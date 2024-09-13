import random
from tqdm import tqdm
from datasets import load_dataset
from evaluator.metrics.base import Evaluator

random.seed(42)

class ARCC(Evaluator):
    def __init__(self, model_name_or_path, tokenizer_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, tokenizer_path, *args, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge')['validation']
        self.pools = load_dataset('allenai/ai2_arc', 'ARC-Challenge')['train']
    
    def get_prompt(self, sample, num_few_shot=4):
        # Select few-shot examples randomly from self.pools
        few_shot_examples = random.sample(list(self.pools), num_few_shot)
        
        # Construct the few-shot part of the prompt
        prompt = "Please answer my question with the correct choices. Here are 4 examples:\n"
        for example in few_shot_examples:
            question = example['question']
            labels = example['choices']['label']
            choices = example['choices']['text']
            prompt += "[INST]\n"
            prompt += f"Question: {question}\n"
            prompt += " [/INST]\n"
            prompt += "Choices:\n"
            for label, choice in zip(labels, choices):
                prompt += f'{label}: {choice}\n'
            prompt += f'Answer:{example["answerKey"]}'
        
        # Add the current sample to the prompt
        question = sample['question']
        labels = sample['choices']['label']
        choices = sample['choices']['text']
        prompt += "[INST]\n"
        prompt += f"Question: {question}\n"
        prompt += " [/INST]\n"
        prompt += "Choices:\n"
        for label, choice in zip(labels, choices):
            prompt += f'{label}: {choice}\n'
        prompt += f'Answer:'
        return prompt
    
    def evaluate(self):
        cnt = 0
        n = len(self.dataset)
        with tqdm(self.dataset) as tbar:
            for sample in tbar:
                prompt = self.get_prompt(sample)
                response = self.generate(prompt)[:8]
                label = sample['answerKey']
                if label in response:
                    cnt += 1
                tbar.set_postfix({'answer': response, 'label': label, 'matched':cnt, 'total': n, 'acc': cnt / n})
        return round(cnt / n, 4)