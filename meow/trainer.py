import torch
from torch import nn
from transformers import Trainer
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker
import torch.nn.functional as F
import copy
import deepspeed
import copy
from pathlib import Path
import datasets

from transformers.utils import is_datasets_available

from meow.utils import get_batch_loss

class ToFUTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    
class ToFUTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.seed = kwargs.pop('seed')
        try:
            self.eval_cfg = kwargs.pop('eval_cfg')
            self.beta = kwargs.pop('beta')
        except:
            pass

        # the coefficient of each part in the loss function. This is used in ablation study.
        if 'npo' in self.loss_type:
            self.npo_coeff = kwargs.pop('npo_coeff')
            self.grad_diff_coeff = kwargs.pop('grad_diff_coeff')
            self.KL_coeff = kwargs.pop('KL_coeff')

            self.ref_policy = kwargs.pop('ref_policy')
        
        super(ToFUTrainerForgetting, self).__init__(*args, **kwargs)
        if "KL" in self.loss_type or 'npo' in self.loss_type or 'dpo' in self.loss_type:
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
        
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')
            
        dataloader_params = {
            'batch_size': self._train_batch_size,
            'collate_fn': data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'persistent_workers': self.args
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params['generator'] = generator
            dataloader_params['shuffle'] = True
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker
            dataloader_params['num_workers'] = 1
        
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss
        
        elif self.loss_type == 'grad_ascent_forgetKL':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = -1  * outputs.loss
            
            with torch.no_grad():
                oracle_outputs = self.oracle_model(
                    input_ids,
                    labels = labels,
                    attention_mask = attention_mask
                )
            oracle_probs = F.log_softmax(oracle_outputs.logits, dim = -1)
            oracle_probs = oracle_probs.view(-1, oracle_outputs.logits.shape[-1])
            current_probs = F.log_softmax(outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, outputs.logits.shape[-1])
            kl_loss = nn.functional.kl_div(
                current_probs,
                oracle_probs,
                reduction='batchmean',
                log_target=True
            )
            loss = forget_loss + kl_loss

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
        
        elif self.loss_type == "grad_ascent_KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        elif self.loss_type in ["dpo", 'dpo_grad_diff', 'dpo_KL']:
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

                idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
                forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean() * 2 / self.beta
            
            if self.loss_type == 'dpo_grad_diff':
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                loss = loss + retain_loss
            
            elif self.loss_type == 'dpo_KL':
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                with torch.no_grad():
                    retain_outputs = self.oracle_model(
                        retain_input_ids,
                        labels = retain_labels,
                        attention_mask = retain_attention_mask
                    )
                retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])
                
                current_outputs = model(retain_input_ids, labels = retain_labels, attention_mask = retain_attention_mask)
                current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
                
                # minimum KL divergence
                retain_loss = nn.functional.kl_div(
                    current_probs,
                    retain_probs,
                    reduction = 'batchmean',
                    log_target = True
                )
                
                loss = loss + retain_loss
            
        elif self.loss_type == 'meow' or self.loss_type == 'meow_batch' or self.loss_type == 'meow_memo':
            retain_input_ids, retain_labels, retain_attn_mask = inputs[0]
            outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attn_mask)
            loss = outputs.loss
            
        elif self.loss_type == 'meow_grad_diff':
            # 取均值
            lose_inputs, win_inputs = inputs
            lose_input_ids, lose_labels, lose_attn_mask = lose_inputs
            outputs = model(lose_input_ids, labels=lose_labels, attention_mask=lose_attn_mask)
            lose_loss = outputs.loss
            lose_loss = lose_loss
            
            win_input_ids, win_labels, win_attn_mask = win_inputs
            outputs = model(win_input_ids, labels=win_labels, attention_mask=win_attn_mask)
            win_loss = outputs.loss
            
            loss = 0.9 * lose_loss + 0.1 * win_loss
        
        elif self.loss_type == 'meow_forgetKL':
            lose_inputs, win_inputs = inputs
            lose_input_ids, lose_labels, lose_attn_mask = lose_inputs
            outputs = model(lose_input_ids, labels=lose_labels, attention_mask=lose_attn_mask)
            lose_loss = outputs.loss
            lose_loss = lose_loss
            
            win_input_ids, win_labels, win_attn_mask = win_inputs
            
            with torch.no_grad():
                oracle_outputs = self.oracle_model(
                    lose_input_ids,
                    labels = lose_labels,
                    attention_mask = lose_attn_mask
                )
            oracle_probs = F.log_softmax(oracle_outputs.logits, dim = -1)
            oracle_probs = oracle_probs.view(-1, oracle_outputs.logits.shape[-1])
            current_probs = F.log_softmax(outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, outputs.logits.shape[-1])
            kl_loss = nn.functional.kl_div(
                current_probs,
                oracle_probs,
                reduction='batchmean',
                log_target=True
            )
            loss = lose_loss + kl_loss
            
        elif self.loss_type == 'meow_dpo':
            lose_inputs, win_inputs = inputs
            lose_input_ids, lose_labels, lose_attention_mask = lose_inputs
            win_input_ids, win_labels, win_attention_mask = win_inputs
            lose_outputs = model(lose_input_ids, labels=lose_labels, attention_mask=lose_attention_mask)
            win_outputs = model(win_input_ids, labels=win_labels, attention_mask=win_attention_mask)
            
            with torch.no_grad():
                lose_outputs_oracle = self.oracle_model(lose_input_ids, labels=lose_labels, attention_mask=lose_attention_mask)
                win_outputs_oracle = self.oracle_model(win_input_ids, labels=win_labels, attention_mask=win_attention_mask)
                lose_logits_oracle = lose_outputs_oracle.logits
                win_logits_oracle = win_outputs_oracle.logits
                
                lose_loss_oracle = -1 * get_batch_loss(lose_logits_oracle, lose_labels)
                win_loss_oracle = -1 * get_batch_loss(win_logits_oracle, win_labels)
            
            win_loss_current = -1 * get_batch_loss(win_outputs.logits, win_labels)
            lose_loss_current = -1 * get_batch_loss(lose_outputs.logits, lose_labels)
            
            pi_logratios = win_loss_current - lose_loss_current
            ref_logratios = win_loss_oracle - lose_loss_oracle
            
            loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean() * 2 / self.beta
            
        elif self.loss_type == 'npo':
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            
            forget_loss_current = get_batch_loss(outputs.logits, labels)
            
            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(
                        input_ids,
                        labels = labels,
                        attention_mask = attention_mask
                    )
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
                           
        elif self.loss_type == 'npo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels)
            
            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
            
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(
                retain_input_ids,
                labels = retain_labels,
                attention_mask = retain_attention_mask
            )
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
            
        elif self.loss_type == 'npo_KL':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels)
            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError

            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            with torch.no_grad():
                retain_outputs = self.oracle_model(
                    retain_input_ids,
                    labels = retain_labels,
                    attention_mask = retain_attention_mask
                )
                
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])
            
            current_outputs = model(retain_input_ids, labels=labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            
            # minimum KL divergence
            retain_loss = nn.functional.kl_div(
                current_probs,
                retain_probs,
                reduction = 'batchmean',
                log_target = True
            )
            loss = self.npo_coeff * forget_loss + self.KL_coeff * retain_loss
        
        elif self.loss_type == 'kto_sigmoid':
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            
            with torch.no_grad():
                idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
                idk_outputs_oracle = self.oracle_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
                idk_loss_log = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
                idk_loss_log_oracle = -1 * get_batch_loss(idk_outputs_oracle.logits, idk_labels)
                
                KL_term = (idk_loss_log - idk_loss_log_oracle).mean()
                
                forget_outputs_oracle = self.oracle_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
                forget_loss_oracle = -1 * get_batch_loss(forget_outputs_oracle.logits, forget_labels)
                
            forget_outputs = model(forget_input_ids, label=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            log_ratios = forget_loss - forget_loss_oracle
            loss = 1.0 - F.sigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta
        
        elif self.loss_type == 'kto_logsigmoid_grad_diff':
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            
            with torch.no_grad():
                idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
                idk_outputs_oracle = self.oracle_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
                idk_loss_log = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
                idk_loss_log_oracle = -1 * get_batch_loss(idk_outputs_oracle.logits, idk_labels)
                
                KL_term = (idk_loss_log - idk_loss_log_oracle).mean()
                
                forget_outputs_oracle = self.oracle_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
                forget_loss_oracle = -1 * get_batch_loss(forget_outputs_oracle.logits, forget_labels)
                
            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            log_ratios = forget_loss - forget_loss_oracle
            forget_loss = 1.0 - F.logsigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta
            print(KL_term)
            
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            
            loss = forget_loss + retain_loss
               
        return (loss, outputs) if return_outputs else loss
        
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)