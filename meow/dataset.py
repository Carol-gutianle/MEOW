import os
import json
import random
import heapq
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
import datasets

from meow.utils import get_model_identifiers_from_yaml, add_dataset_index
from meow.utils import load_data_from_json, save_data_to_json, split_paragraph

'''
ToFU相关Dataset
'''
def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def custom_data_collator_forget(samples):
    rets = []
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        data_types = ["idk", "forget", "retain"]
    elif len(samples[0]) == 2:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        data_types = ["forget", "retain"]
    elif len(samples[0]) == 1:
        idk_samples = [sample[0] for sample in samples]
        data_types = ['idk']
    for data_type in data_types:
        if data_type == "forget":
            data = forget_samples 
        elif data_type == "retain":
            data = retain_samples 
        elif data_type == "idk":
            data = idk_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def meow_data_collator_forget(samples):
    rets = []
    meow_samples = []
    for sample in samples:
        for sa in sample:
            meow_samples.append(sa)
    input_ids = [s[0] for s in meow_samples]
    labels = [s[1] for s in meow_samples]
    attention_mask = [s[2] for s in meow_samples]
    rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def meow_data_collator_forget_double(samples):
    rets = []
    forget_samples = []
    retain_samples = []
    for sample in samples:
        n = len(sample)
        for i in range(n):
            if i < n // 2:
                forget_samples.append(sample[i])
            else:
                retain_samples.append(sample[i])
    forget_input_ids = [s[0] for s in forget_samples]
    forget_labels = [s[1] for s in forget_samples]
    forget_attention_mask = [s[2] for s in forget_samples]
    retain_input_ids = [s[0] for s in retain_samples]
    retain_labels = [s[1] for s in retain_samples]
    retain_attention_mask = [s[2] for s in retain_samples]
    rets.append((torch.stack(forget_input_ids), torch.stack(forget_labels), torch.stack(forget_attention_mask)))
    rets.append((torch.stack(retain_input_ids), torch.stack(retain_labels), torch.stack(retain_attention_mask)))
    return rets

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets
    
class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)
                
class TextForgetDatasetMeow(Dataset):
    def __init__(self, forget_data_path, tokenizer, model_family, max_length=512, loss_type='meow', split='forget10', top_k=3, num_batch=8, max_rouge = 1.0, metric='smallest') -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = load_data_from_json(forget_data_path)
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data =datasets.load_dataset('locuslab/TOFU', retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        self.top_k = top_k
        self.num_batch = num_batch
        self.max_rouge = max_rouge
        self.metric = metric
        if loss_type == 'meow':
            self.splits = ['forget']
        elif loss_type == 'meow_batch':
            self.splits = ['meow_batch']
        elif loss_type == 'meow_memo':
            self.splits = ['meow_memo']
        elif loss_type in ['meow_forgetKL', 'meow_grad_diff']:
            self.splits = ['meow_memo', 'retain']
        super().__init__()
    def __len__(self):
        return len(self.forget_data)
    def __getitem__(self, idx):
        rets = []
        for data_type in self.splits:
            forget_data = self.forget_data
            question = forget_data[idx]['question']
            if data_type == 'meow_batch':
                for i in range(self.num_batch):
                    try:
                        answer = forget_data[idx][f'answer_{i}']
                        converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                        rets.append(converted_data)
                    except:
                        continue
                if len(rets) < self.num_batch:
                    for i in range(self.num_batch - len(rets)):
                        rets.append(rets[0])
            elif data_type == 'meow_memo':
                matched_keys = []
                for key in forget_data[idx].keys():
                    if key.startswith('answer') and key.endswith('rouge1'):
                        matched_keys.append(key)
                # generate_candidate_data
                if self.metric == 'smallest':
                    top_values = heapq.nsmallest(self.top_k, (forget_data[idx][key] for key in matched_keys))
                elif self.metric == 'largest':
                    top_values = heapq.nlargest(self.top_k, (forget_data[idx][key] for key in matched_keys))
                elif self.metric == 'random':
                    random.seed(42)
                    candidates = [forget_data[idx][key] for key in matched_keys]
                    if len(candidates) < self.top_k:
                        candidates += [forget_data[idx]['answer_0_rouge1']] * (self.top_k - len(candidates))
                    top_values = random.sample(candidates, self.top_k)
                else:
                    raise ValueError(0)
                top_keys = []
                for value in top_values:
                    for key in matched_keys:
                        if forget_data[idx][key] == value:
                            top_keys.append(key)
                            break
                for i in range(self.num_batch):
                    try:
                        answer = forget_data[idx][top_keys[i % self.top_k].replace('_rouge1', '')]
                    except:
                        answer = forget_data[idx]['answer_0']
                    converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                    rets.append(converted_data)
            elif data_type == 'retain':
                data = self.retain_data
                question = data[idx]['question']
                answer = data[idx]['answer']
                for i in range(self.num_batch):
                    converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                    rets.append(converted_data)
        return rets
    
class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.idontknowfile = "idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets