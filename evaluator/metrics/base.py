import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from meow.message import Robot

class Evaluator:
    def __init__(self, model_name_or_path, tokenizer_path, *args, **kwargs) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto"
        )
        self.robot = Robot()
        
    def get_prompt(self, sample):
        pass
        
    def generate(self, prompt):
        input_ids = self.tokenizer(
            prompt, 
            return_tensors = 'pt'
        )['input_ids'].to(self.device)
        generate_ids = self.model.generate(
            input_ids = input_ids,
            return_dict_in_generate = True,
            max_new_tokens = 128,
            do_sample = False,
            pad_token_id = self.tokenizer.eos_token_id
        )
        output = self.tokenizer.decode(
            generate_ids.sequences[0]
        ).replace(prompt, '')
        return output
    
    def evaluate(self):
        pass