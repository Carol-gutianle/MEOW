import argparse
from meow.message import Robot
from evaluator.metrics.arcc import ARCC
from evaluator.metrics.arce import ARCE
from evaluator.metrics.piqa import PIQA

class NLU:
    
    def __init__(self, model_name_or_path, tokenizer_path):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_path = tokenizer_path
        self.robot = Robot()

    def evaluate(self, split):
        if split == 'arcc':
            arcc = ARCC(self.model_name_or_path, self.tokenizer_path)
            arcc_score = arcc.evaluate()
            message = f'ARCC Finished! Model Path:{self.model_name_or_path}, Acc: {arcc_score}!'
        elif split == 'arce':
            arce = ARCE(self.model_name_or_path, self.tokenizer_path)
            arce_score = arce.evaluate()
            message = f'ARCE Finished! Model Path:{self.model_name_or_path}, Acc: {arce_score}!'
        elif split == 'piqa':
            piqa = PIQA(self.model_name_or_path, self.tokenizer_path)
            piqa_score = piqa.evaluate()
            message = f'PIQA Finished! Model Path:{self.model_name_or_path}, Acc: {piqa_score}!'
        elif split == 'all':
            arcc = ARCC(self.model_name_or_path, self.tokenizer_path)
            arcc_score = arcc.evaluate()
            arce = ARCE(self.model_name_or_path, self.tokenizer_path)
            arce_score = arce.evaluate()
            piqa = PIQA(self.model_name_or_path, self.tokenizer_path)
            piqa_score = piqa.evaluate()
            message = f'NLU Test Finished! Model Path:{self.model_name_or_path}, PIQA:{piqa_score}, ARCC:{arcc_score}, ARCE:{arce_score}!'
        print(message)
        self.robot.post_message(message)
        
def main():
    
    parser = argparse.ArgumentParser(description='Script for NLG Testing.')
    parser.add_argument('--split', choices=['all', 'arcc', 'arce', 'piqa'])
    parser.add_argument('--model_name_or_path', type=str, default='Path to Model')
    parser.add_argument('--tokenizer_path', type=str, default='PATH TO tokenizer')
    args = parser.parse_args()
    
    nlu = NLU(args.model_name_or_path, args.tokenizer_path)
    nlu.evaluate(args.metric)
    
if __name__ == "__main__":
    main()