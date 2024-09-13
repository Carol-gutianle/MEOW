import os
import argparse
import random
import torch
import mauve
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
from meow.message import Robot
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)

class NLG:
    def __init__(self, model_name_or_path, tokenizer_path):
        self.cc_news = load_dataset('vblagoje/cc_news')['train']
        self.wikipedia = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')['train']
        self.combined_data = self.cc_news['text'] + self.wikipedia['text']
        self.filtered_data = [text for text in self.combined_data if len(text) > 100 and len(text) < 200]
        self.data = random.sample(self.filtered_data, 5000)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
        self.model_name_or_path = model_name_or_path
        self.robot = Robot()
    
    def cal_rep3(self):
        total_rep3 = 0.0
        
        with tqdm(self.data) as tbar:
            for sample in tbar:
                input_ids = self.tokenizer(
                    sample,
                    return_tensors = 'pt'
                )['input_ids'][0].to(self.device)
                
                generate_ids = self.model.generate(input_ids = input_ids[:32].unsqueeze(0),return_dict_in_generate = True,max_new_tokens = max(1, len(input_ids) - 32), pad_token_id=self.tokenizer.eos_token_id)
                
                output = self.tokenizer.decode(
                    generate_ids.sequences[0]
                )
                        
                words = output.split()
                if len(words) < 3:
                    return 0.0
                trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
                total_trigrams = len(trigrams)
                unique_trigrams = len(set(trigrams))
                rep3 = 1 - (unique_trigrams / total_trigrams)
                total_rep3 += rep3
                tbar.set_postfix({'Rep3': rep3, 'Total Rep3': total_rep3})
        
        return total_rep3 / len(self.final_data)
                
    
    def cal_mauve(self):
        generated_texts = []
        references = []
        with tqdm(self.data) as tbar:
            for sample in tbar:
                references.append(sample)
                input_ids = self.tokenizer(
                    sample,
                    return_tensors = 'pt'
                )['input_ids'][0].to(self.device)
                
                generate_ids = self.model.generate(input_ids = input_ids[:32].unsqueeze(0),return_dict_in_generate = True,max_new_tokens = max(1, len(input_ids) - 32), pad_token_id=self.tokenizer.eos_token_id)
                
                output = self.tokenizer.decode(
                    generate_ids.sequences[0]
                )

                generated_texts.append(output)

        out = mauve.compute_mauve(
            p_text = generated_texts,
            q_text = references,
            device_id = 0,
            mauve_scaling_factor = 2,
            max_text_length = 256,
            verbose = False
        )
        
        return out.mauve
    
    def cal_bleu(self):
        references = []
        outputs = []
        
        with tqdm(self.data) as tbar:
            for sample in tbar:
                input_ids = self.tokenizer(
                    sample,
                    return_tensors = 'pt'
                )['input_ids'][0].to(self.device)
                
                generate_ids = self.model.generate(input_ids = input_ids[:32].unsqueeze(0),return_dict_in_generate = True,max_new_tokens = max(1, len(input_ids) - 32), pad_token_id=self.tokenizer.eos_token_id)
                
                output = self.tokenizer.decode(
                    generate_ids.sequences[0]
                )
                
                references.append(sample)
                outputs.append(output)
        
        bleu = load("bleu")
        results = bleu.compute(predictions=outputs, references=references)
        return results['bleu']
        
    
    def evaluate(self, metric='all'):
        
        print(f'Current Model: {self.model_name_or_path}!')
        
        if metric == 'mauve':
            mauve_score = self.cal_mauve()
            message = f'MAUVE Finished! Model Path:{self.model_name_or_path}, MAUVE: {mauve_score}!'
        elif metric == 'rep3':
            rep3 = self.cal_rep3()
            message = f'Rep3 Finished! Model Path:{self.model_name_or_path}, Rep3: {rep3}!'
        elif metric == 'bleu':
            bleu = self.cal_bleu()
            message = f'BLEU Finished! Model Path:{self.model_name_or_path}, BLEU: {bleu}!'
        elif metric == 'all':
            mauve_score = self.cal_mauve()
            ppl_score = self.cal_ppl()
            rep3 = self.cal_rep3()
            message = f'NLG Test Finished! Model Path: {self.model_name_or_path}, MAUVE: {mauve_score}, PPL: {ppl_score}, Rep3: {rep3}!'
        else:
            raise NotImplementedError
        
        print(message)
        self.robot.post_message(message)
        

def main():
    
    parser = argparse.ArgumentParser(description='Script for NLG Testing.')
    parser.add_argument('--metric', choices=['all', 'mauve', 'rep3', 'bleu'])
    parser.add_argument('--model_name_or_path', type=str, default='Path to Model')
    parser.add_argument('--tokenizer_path', type=str, default='PATH TO tokenizer')
    args = parser.parse_args()
    
    nlg = NLG(args.model_name_or_path, args.tokenizer_path)
    nlg.evaluate(args.metric)
    
if __name__ == "__main__":
    main()