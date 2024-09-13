import re
import json
import argparse

import json

def load_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_data_to_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def remove_empty_answer(data, save_path):
    removed_data = []
    pattern = r'^answer_\d+$'
    for sample in data:
        removed_sample = {}
        removed_sample['question'] = sample['question']
        removed_sample['undesired_answer'] = sample['undesired_answer']
        answer_keys = [key for key in sample.keys() if re.match(pattern, key)]
        i = 0
        for answer_key in answer_keys:
            if len(sample[answer_key]) != 0:
                removed_sample[f'answer_{i}'] = sample[answer_key]
                i += 1
        removed_data.append(removed_sample)
    save_data_to_json(save_path, removed_data)
    
def reverse_answer(data, save_path):
    reversed_answers = []
    for sample in data:
        reversed_answer = sample.copy()
        undesired_answer = sample['undesired_answer']
        words = undesired_answer.split(' ')
        reversed_words = words[::-1]
        reversed_answer['reversed_answer'] = ' '.join(reversed_words)
        reversed_answers.append(reversed_answer)
    save_data_to_json(save_path, reversed_answers)
    
def convert_answer(data, save_path):
    converted_answers = []
    for sample in data:
        converted_answer = {
            'question': sample['question']
        }
        for j in range(4):
            try:
                converted_answer['answer'] = sample[f'answer_{j}']
                converted_answers.append(converted_answer.copy())
            except:
                continue
    save_data_to_json(save_path, converted_answers)
    
def generate_dpo_answer(data, save_path):
    dpo_answers = []
    for sample in data:
        dpo_answer = {
            'question': sample['question'],
            'undesired_answer': sample['undesired_answer']
        }
        for j in range(4):
            try:
                dpo_answer['answer'] = sample[f'answer_{j}']
                dpo_answers.append(dpo_answer.copy())
            except:
                continue
    save_data_to_json(save_path, dpo_answers)
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for augmentation and convert data')
    parser.add_argument('--mode', choices=['aug', 'remove', 'reverse', 'convert', 'dpo'], help='Choose the mode you need.')
    parser.add_argument('--split', type=str, default='forget05_perturbed')
    args = parser.parse_args()
    return args
        
def main():
    args = parse_arguments()
    if args.mode == 'remove':
        data = load_data_from_json(f'{args.split}.json')
        remove_empty_answer(data, f'{args.split}_removed.json')
    elif args.mode == 'reverse':
        data = load_data_from_json(f'{args.split}_removed.json')
        reverse_answer(data, f'{args.split}_reversed.json')
    elif args.mode == 'convert':
        data = load_data_from_json(f'{args.split}_removed.json')
        convert_answer(data, f'{args.split}_converted.json')
    elif args.mode == 'dpo':
        data = load_data_from_json(f'{args.split}_removed.json')
        generate_dpo_answer(data, f'{args.split}_dpo.json')
        
        
if __name__ == "__main__":
    main()