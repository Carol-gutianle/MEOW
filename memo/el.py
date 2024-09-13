import os
import torch
from tqdm import tqdm
from copy import deepcopy
from meow.utils import load_data_from_json, save_data_to_json
from transformers import AutoModelForCausalLM, AutoTokenizer

class EL:
    def __init__(self, data_path, save_data_path, model_path, tokenizer_path) -> None:
        self.data = load_data_from_json(data_path)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.save_data_path = save_data_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model.to(self.device).eval()
    
    def calculate_el(self, question, answer, mode):
        
        def word_ngrams(text, n):
            words = text.split()
            ngrams = [''.join(words[i:i+n]) for i in range(len(words)-n+1)]
            return ngrams
        
        def overlap(n, text_a, text_b):
            n_gram_a = set(word_ngrams(text_a, n))
            n_gram_b = set(word_ngrams(text_b, n))
            n_a = len(n_gram_a)
            cnt = 0
            for c in n_gram_a:
                if c in n_gram_b:
                    cnt += 1
            return cnt / n_a
        
        def el(n, text):
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            n_ids = len(input_ids)
            sum_overlap = 0
            cnt = 0
            for i in range(1, n_ids - n):
                curr_ids = input_ids[:i]
                curr_input = self.tokenizer.decode(curr_ids)
                curr_input_ids = self.tokenizer(curr_input, return_tensors='pt')['input_ids'].to(self.device)
                generate_ids = self.model.generate(input_ids=curr_input_ids, return_dict_in_generate=True, max_new_tokens=200)
                output = self.tokenizer.decode(generate_ids.sequences[0]).replace(curr_input, '')
                ol = overlap(n, output, text.replace(curr_input, ''))
                sum_overlap += ol
                cnt += 1
            return sum_overlap / cnt 

        def el_prefix(n, question, answer):
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            n_ids = len(answer_ids)
            sum_overlap = 0
            cnt = 0
            for i in range(1, n_ids - n):
                curr_ids = answer_ids[:i]
                curr_input = self.tokenizer.decode(curr_ids)
                text = f'{question} {curr_input}'
                curr_input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
                generate_ids = self.model.generate(input_ids=curr_input_ids, return_dict_in_generate=True, max_new_tokens=200)
                output = self.tokenizer.decode(generate_ids.sequences[0]).replace(text, '')
                ol = overlap(n, output, answer)
                sum_overlap += ol
                cnt += 1
            return sum_overlap / cnt
        
        n = 1
        text = f'{question} {answer}'
        if mode == 'prefix':
            return  el(n, text)
        elif mode == 'suffix':
            return el_prefix(n, question, answer)
    
    def run(self):
        el_list = []
        with tqdm(self.data) as tbar:
            for sample in tbar:
                updated_sample = deepcopy(sample)
                keys = sample.keys()
                for key in keys:
                    if key == 'question':
                        continue
                    else:
                        updated_sample[f'{key}_el'] = self.calculate_el(sample['question'], sample[key])
                el_list.append(updated_sample)
        save_data_to_json(os.path.join(self.save_data_path, 'el_forget05_full.json'), el_list)