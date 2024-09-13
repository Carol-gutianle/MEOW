import os
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, TrainingArguments

from meow.trainer import ToFUTrainerForgetting
from meow.dataset import (
    TextForgetDatasetQA,
    TextForgetDatasetMeow,
    TextForgetDatasetDPOQA
)
from meow.dataset import custom_data_collator_forget, meow_data_collator_forget, meow_data_collator_forget_double
from meow.utils import get_model_identifiers_from_yaml, EarlyStoppingCallback, load_data_from_json
from meow.message import Robot

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(config_path='config', config_name='forget_phi_ga')
def main(args):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    set_seed(args.seed)
    
    # if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    
    model_args = get_model_identifiers_from_yaml(args.model_family)
    model_id = model_args['hf_key']
    
    if args.model_path is None:
        args.model_path = model_args['ft_model_path']

    
    if local_rank == 0:
        if os.path.exists(args.save_dir):
            print('Directory already exists.')
            if not args.overwrite_dir:
                exit()
    
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        with open(f'{args.save_dir}/config.yaml', 'w') as file:
            OmegaConf.save(args, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    max_length = args.max_length
    
    # 修改 forget loss
    if args.forget_loss in ['dpo', 'dpo_KL', 'dpo_grad_diff']:
        torch_format_dataset = TextForgetDatasetDPOQA(
            args.data_path,
            tokenizer = tokenizer,
            model_family = args.model_family,
            max_length = max_length,
            split = args.split
        )
    elif 'meow' in args.forget_loss:
        torch_format_dataset = TextForgetDatasetMeow(
            args.data_path,
            tokenizer = tokenizer,
            model_family = args.model_family,
            max_length = max_length,
            loss_type = args.forget_loss,
            top_k = args.top_k,
            split = args.split,
            metric = args.metric
        )
    else:
        torch_format_dataset = TextForgetDatasetQA(
            args.data_path, 
            tokenizer = tokenizer, 
            model_family = args.model_family, 
            max_length=max_length, 
            split = args.split, 
            loss_type = args.forget_loss
        )
    
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)
    
    if args.max_steps != -1:
        max_steps = args.max_steps
    else:
        max_steps = int(args.num_epochs * len(torch_format_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    print('最大步数', max_steps)
    
    training_args = TrainingArguments(
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = max(1, steps_per_epoch),
        max_steps = max_steps,
        learning_rate = args.lr,
        bf16 = True,
        bf16_full_eval = True,
        logging_steps = max(1, max_steps // 20),
        logging_dir = f'{args.save_dir}/logs',
        output_dir = args.save_dir,
        optim = 'paged_adamw_32bit',
        save_strategy = 'steps' if args.save_model and (not args.eval_only) else 'no',
        save_steps = max_steps,
        save_only_model = True,
        ddp_find_unused_parameters = False,
        deepspeed = 'config/ds_config.json',
        weight_decay = args.weight_decay,
        eval_steps = steps_per_epoch,
        evaluation_strategy = 'steps' if args.eval_while_train else "no",
        seed = args.seed
    )
    
    import re
    path_found = False
    for file in os.listdir(args.model_path):
        if re.search('pytorch.*\.bin', file):
            path_found = True
            break
        if re.search('model-*\.safetensors', file):
            path_found = True
            break
    
    oracle_model = None
    
    # 获取模型
    if path_found:
        config = AutoConfig.from_pretrained(model_id)
        print('Loading from checkpoint...')
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config = config,
            use_flash_attention_2 = model_args['flash_attention2'] == 'true',
            torch_dtype = torch.bfloat16,
            trust_remote_code = True
        )
        if ('KL' in args.forget_loss) or ('npo' in args.forget_loss) or ('dpo' in args.forget_loss):
            oracle_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                config = config,
                use_flash_attention_2 = model_args['flash_attention2'] == 'true',
                torch_dtype = torch.bfloat16,
                trust_remote_code = True
            )
    else:
        print('Loading from checkpoint...')
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_flash_attention_2 = model_args['flash_attention2'] == 'true',
            torch_dtype = torch.bfloat16,
            device_map = device_map
        )
        model = PeftModel.from_pretrained(model, model_id = args.model_path)
        model = model.merge_and_unload()
        model.save_pretrained(args.model_path)
    model.generation_config.do_sample = True
    
    if model_args['gradient_checkpointing'] == 'true':
        model.gradient_checkpointing_enable()
    
    config = LoraConfig(
        r = args.LoRA.r,
        lora_alpha = args.LoRA.alpha,
        target_modules = find_all_linear_names(model),
        lora_dropout = args.LoRA.dropout,
        bias = 'none',
        task_type = 'CAUSAL_LM'
    )
    
    if args.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)
        
    if args.forget_loss == 'npo' or args.forget_loss == 'dpo':
        loss_threshold = 0
        
    early_stopping_callback = EarlyStoppingCallback(loss_threshold=0.01)
    
    # set collate
    if 'meow' not in args.forget_loss:
        collate = custom_data_collator_forget
    else:
        collate = meow_data_collator_forget_double
    
    if 'npo' not in args.forget_loss:
        trainer = ToFUTrainerForgetting(
            model = model,
            tokenizer = tokenizer,
            train_dataset = torch_format_dataset,
            eval_dataset = torch_format_dataset,
            # callbacks = [early_stopping_callback],
            compute_metrics = None,
            args = training_args,
            data_collator = collate,
            oracle_model = oracle_model,
            forget_loss = args.forget_loss,
            eval_cfg = args.eval,
            seed = args.seed,
            beta = args.beta
        )
    else:
        trainer = ToFUTrainerForgetting(
            model=model,
            tokenizer=tokenizer,
            train_dataset = torch_format_dataset,
            eval_dataset = torch_format_dataset,
            compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
            # callbacks=[GlobalStepDeletionCallback],
            args=training_args,
            data_collator = collate,
            oracle_model = oracle_model,
            forget_loss = args.forget_loss,
            eval_cfg = args.eval,
            seed = args.seed,
            ref_policy = args.ref_policy,
            beta = args.beta,
            npo_coeff=args.npo_coeff,
            grad_diff_coeff=args.grad_diff_coeff,
            KL_coeff=args.KL_coeff
        )    
        
    model.config.use_cache = False
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()
        
    if args.save_model and (not args.eval_only):
        print('保存')
        print(f'Forget后的模型存放于{args.save_dir}')
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        
    if local_rank == 0:
        robot = Robot()
        robot.post_message(f'forget实验完成, 存储到{args.save_dir}/checkpoint-{max_steps}, 参数为maxlength-{args.max_length}-top{args.top_k}-metric{args.metric}!')
    
if __name__ == "__main__":
    try:
        main()
    except:
        robot = Robot()
        robot.post_message('额..有个实验失败了...')