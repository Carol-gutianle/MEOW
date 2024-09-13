import os
import torch
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from meow.utils import load_data_from_json, save_data_to_json

class MA:
    def __init__(self, data_path, save_data_path, model_path, tokenizer_path):
        self.data = load_data_from_json(data_path)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.save_data_path = save_data_path
        self.model.to(self.device).eval()
    
    def calculate_ma(self, question, answer, mode):
        
        def ma(text):
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            n_ids = len(input_ids)
            acc = 0
            cnt = 0
            for i in range(1, n_ids):
                cnt += 1
                curr_ids = input_ids[:i]
                next_token = input_ids[i]
                curr_input = self.tokenizer.decode(curr_ids)
                curr_input_ids = self.tokenizer(curr_input, return_tensors='pt')['input_ids'].to(self.device)
                predict_next_token = self.model.generate(input_ids=curr_input_ids, return_dict_in_generate=True, max_new_tokens=1)['sequences'][0][-1].item()
                if predict_next_token == next_token:
                    acc += 1
            return acc / cnt
            
        def ma_prefix(question, answer):
            # 冻结输入
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            n_ids = len(answer_ids)
            acc = 0
            cnt = 0
            for i in range(1, n_ids):
                cnt += 1
                curr_ids = answer_ids[:i]
                next_token = answer_ids[i]
                curr_answer = self.tokenizer.decode(curr_ids)
                text = f'{question} {curr_answer}'
                curr_input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
                predict_next_token = self.model.generate(input_ids=curr_input_ids, return_dict_in_generate=True, max_new_tokens=1)['sequences'][0][-1].item()
                if predict_next_token == next_token:
                    acc += 1
            return acc / cnt
        
        text = f'{question} {answer}'
        if mode == 'prefix':
            return ma(text)
        elif mode == 'suffix':
            return ma_prefix(question, answer)
        
        return ma(text)
    
    def run(self, mode):
        el_list = []
        with tqdm(self.data) as tbar:
            for sample in tbar:
                updated_sample = deepcopy(sample)
                keys = sample.keys()
                for key in keys:
                    if key == 'question':
                        continue
                    else:
                        updated_sample[f'{key}_ma'] = self.calculate_ma(sample['question'], sample[key], mode)
                el_list.append(updated_sample)
        save_data_to_json(os.path.join(self.save_data_path, 'ma_full_forget05.json'), el_list)