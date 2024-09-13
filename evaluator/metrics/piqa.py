from tqdm import tqdm
from evaluator.metrics.base import Evaluator
from evaluator.metrics.utils import read_jsonlines, read_lst


class PIQA(Evaluator):
    def __init__(self, model_name_or_path, tokenizer_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, tokenizer_path, *args, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.questions = read_jsonlines('piqa/dev.jsonl')
        self.labels = read_lst('piqa/dev-labels.lst')
        self.train_questions = read_jsonlines('piqa/train.jsonl')[:4]
        self.train_labels = read_lst('piqa/train-labels.lst')[:4]
         
    def evaluate(self):
        cnt = 0
        n = len(self.questions)
        with tqdm(zip(self.questions, self.labels), total=len(self.questions)) as tbar:
            for data, label in tbar:
                question = data['goal']
                sol1 = data['sol1']
                sol2 = data['sol2']
                prompt = f'Here are some examples:'
                for i in range(4):
                    prompt += f'Example {i+1}: Here is a question: "{self.train_questions[i]["goal"]}". Please answer with 0 or 1 based on the following solutions:\n 0. {self.train_questions[i]["sol1"]} \n1. {self.train_questions[i]["sol2"]}\n Answer: {self.train_labels[i]}\n'
                prompt += f'Here is a question: "{question}". Please answer with 0 or 1 based on the following solutions:\n 0. {sol1} \n1. {sol2}\n Answer:'
                answer = self.generate(prompt)[:10]
                if str(label) in answer:
                    cnt += 1
                tbar.set_postfix({'answer': answer, 'matched':cnt, 'total': n, 'acc': cnt / n})
        return round(cnt / n, 4)