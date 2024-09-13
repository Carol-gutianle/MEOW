import os
from copy import deepcopy
import random
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from dataclasses import dataclass
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

import json

def seed_all(seed = 8888):
    torch.manual_seed(seed)
    random.seed(seed)
    

def load_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_data_to_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        
@dataclass
class memo_rouger:
    rouge1: 0
    rouge2: 0
    rougeL: 0
    total: 0
    
    def update(self, scorer):
        self.rouge1 += scorer['rouge1'].precision
        self.rouge2 += scorer['rouge2'].precision
        self.rougeL += scorer['rougeL'].precision
        
    def get_average(self):
        try:
            self.rouge1 /= self.total
            self.rouge2 /= self.total
            self.rougeL /= self.total
        except:
            self.rouge1 = 0.0
            self.rouge2 = 0.0
            self.rougeL = 0.0
        
    def get_value(self):
        return {
            'rouge1': self.rouge1,
            'rouge2': self.rouge2,
            'rougeL': self.rougeL
        }
    
    def get_rouge1(self, prefix=''):
        return {
            f'{prefix}_rouge1': self.rouge1
        }
        

class MEMO:
    def __init__(self, data_path, save_data_path, model_path, tokenizer_path) -> None:
        self.save_data_path = save_data_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.data_path = data_path
        if data_path.endswith('.json'):
            self.raw_data = load_data_from_json(data_path)
        else:
            self.raw_data = pd.read_csv(data_path).to_dict('records')
        # set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model.to(self.device).eval()

    def cal_rouge(self, prompt, answer):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # 使用Greedy Search计算标准答案
        input_ids = self.tokenizer(
            prompt,
            return_tensors = 'pt'
        )['input_ids'].to(self.device)
        generate_ids = self.model.generate(
            input_ids = input_ids,
            return_dict_in_generate = True,
            max_new_tokens = 200
        )
        output = self.tokenizer.decode(
            generate_ids.sequences[0]
        ).replace(prompt, '')
        scores = scorer.score(answer, output)
        return scores
    
    def split(self, sliding_length, question, answer, mode='prefix'):
        # mode: prefix or suffix
        list_of_substrs = []
        max_question_length = len(question)
        max_answer_length = len(answer)
        if mode == 'prefix':
            for sub_length in range(1, max_question_length, sliding_length):
                return_dict = {
                    'subquestion': question[:sub_length],
                    'label': question[sub_length:] + ' ' + answer
                }
                list_of_substrs.append(return_dict)
        elif mode == 'suffix':
            for sub_length in range(1, max_answer_length, sliding_length):
                return_dict = {
                    'subquestion': question + ' ' + answer[:sub_length],
                    'label': answer[sub_length:]
                }
                list_of_substrs.append(return_dict)
        return list_of_substrs   
        
    def run(self, mode):
        # 生成完整扰乱数据集的rouge
        data = self.raw_data
        total_data = []
        sliding_length = 5
        with tqdm(data) as tbar:
            for sample in data:
                updated_sample = deepcopy(sample)
                question = sample['question']
                keys = sample.keys()
                for i, key in enumerate(keys):
                    print(i)
                    if key == 'question':
                        continue
                    subquestions = self.split(sliding_length, question, sample[key], mode)
                    cnt = len(subquestions)
                    rouger = memo_rouger(0.0, 0.0, 0.0, cnt)
                    for subquestion in subquestions:
                        rouge = self.cal_rouge(subquestion['subquestion'], subquestion['label'])
                        rouger.update(rouge)
                    rouger.get_average()
                    score = rouger.get_rouge1(key)
                    updated_sample.update(score)
                tbar.update(1)
                tbar.set_postfix(score)
                total_data.append(updated_sample)
        save_data_path = os.path.basename(self.data_path).replace('.json', '') + f'_memo_{mode}_{sliding_length}_full.json'
        save_data_to_json(save_data_path, total_data)    
        
    def draw(self, data_path, title='default'):
        # 记忆程度可视化
        data = load_data_from_json(data_path)
        data = pd.DataFrame(data)
        memo = data['rouge1'].values
        self.analyze(memo, title)
        idxes = [i for i in range(len(data))]
        plt.figure(figsize=(12, 6))
        plt.bar(idxes, data['rouge1'])
        plt.ylabel('Memorization')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.show()
        
    def analyze(self, memo, title='default'):
        variance = np.var(memo)
        std_dev = np.std(memo)
        mean = np.mean(memo)
        cv = std_dev / mean
        print(f'Model:\t{title}')
        print(f'Variance:\t{np.round(variance, 4)}')
        print(f'Mean:\t{np.round(mean, 4)}')
        print(f'STD_DEV:\t{np.round(std_dev, 4)}')
        print(f'CV:\t{np.round(cv, 4)}')
        print(f'Max:\t{np.round(np.max(memo), 4)}')
        print(f'Min:\t{np.round(np.min(memo), 4)}')
        
    def draw_memo(self):
        models = ['llama2', 'llama13b', 'phi', 'pythia']
        memo_result = pd.DataFrame([])
        for model in models:
            target_path = os.path.join(self.save_data_path, f'{model}-forget05_perturbed_generated_memo_suffix_5.json')
            data = pd.DataFrame(load_data_from_json(target_path))
            memo_result = pd.concat([memo_result, data['rouge1']], axis=1)
        memo_result.to_csv('memo_result.csv', index=None)
            
        
    def corr(self, data_path, save_path):
        import seaborn as sns
        # 对三个指标进行相关性分析
        data = pd.DataFrame(load_data_from_json(data_path))
        memo = data[['rouge1', 'rouge2', 'rougeL']]
        corr = memo.corr()
        sns.heatmap(corr, annot=True, cmap='OrRd', center=0)
        plt.savefig(os.path.join(self.save_path, save_path))
        plt.show()