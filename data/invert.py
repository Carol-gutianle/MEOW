# 进行数据增广
import os
import time
import json
import httpx
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
from meow.utils import load_data_from_json, save_data_to_json

os.environ['OPENAI_API_KEY'] = 'API_KEY'
os.environ['OPENAI_BASE_URL'] = 'BASE_URL'

NUM_GENERATED = 4

class LLM:
    
    def __init__(self, engine="gpt-4-1106-preview", temperature=0.9, sleep_time=5) -> None:
        self.engine = engine
        self.client = OpenAI(
            api_key = os.environ.get('OPENAI_API_KEY'),
            base_url = os.environ.get('OPENAI_BASE_URL'),
            http_client = httpx.Client(
                base_url = os.environ.get('OPENAI_BASE_URL'),
                follow_redirects = True
            )
        )
        self.temperature = temperature
        self.sleep_time = sleep_time
    
    def call(self, prompts):
        status = 0
        while status != 1:
            try:
                completion = self.client.chat.completions.create(
                    model = self.engine,
                    messages = prompts,
                    temperature = self.temperature,
                    max_tokens = 512
                )
                RESPONSE = completion.choices[0].message.content
                status = 1
                time.sleep(self.sleep_time)
            except Exception as e:
                print(e)
                time.sleep(5)
                pass
        return RESPONSE

class Invert:
    
    def __init__(self) -> None:
        self.llm = LLM()
        
    def fact_invert_tofu(self, split):
        data_list = self.get_forget_data(split)
        rewrite_answer = []
        for data in tqdm(data_list):
            try:
                message = [
                    {
                        'role': 'system',
                        'content': f'Please generate {NUM_GENERATED} answers based on the Question and Answer that do not factually match the Answer. Please respond with each answer on a separate line, without adding any numbers or extraneous markers.'
                    },
                    {
                        'role': 'user',
                        'content': f'''Question: {data['question']} Answer: {data['undesired_answer']}'''
                    }
                ]
                response = self.llm.call(message).split('\n')
                for i, res in enumerate(response):
                    data[f'answer_{i}'] = res
                rewrite_answer.append(data)
            except:
                with open(f'{split}_part{len(rewrite_answer)}.json', 'w') as f:
                    json.dump(rewrite_answer, f, indent=4)
        with open(f'{split}_{NUM_GENERATED}.json', 'w') as f:
            json.dump(rewrite_answer, f, indent=4)
            
    def get_forget_data(self, split=None, dataset='tofu'):
        data_list = []
        if dataset == 'tofu':
            forget_question = load_dataset('locuslab/TOFU', split)['train']['question']
            forget_answer = load_dataset('locuslab/TOFU', split)['train']['answer']
            for q, a in zip(forget_question, forget_answer):
                data_list.append(
                    {
                        'question': q,
                        'undesired_answer': a
                    }
                )
        return data_list
   
aug = Invert()
aug.fact_invert_tofu('forget05')