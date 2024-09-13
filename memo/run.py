import argparse
import time
from memo.el import EL
from memo.ma import MA
from memo.memo import MEMO

from meow.message import Robot

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for quantifying memorization in LLMs.')
    parser.add_argument('--metric', choices=['memo','el','ma'])
    parser.add_argument('--mode', choices=['prefix', 'suffix'])
    parser.add_argument('--data_path', type=str, default='forget_perturbed_generated.json', help='PATH to inverted facts')
    parser.add_argument('--save_path', type=str, default='../data/memo_data', help="PATH TO save memorization data.")
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    data_path = args.data_path
    save_path = args.save_path
    model_name_or_path = args.model_name_or_path
    tokenizer_path = args.tokenizer_path
    
    robot = Robot()
    
    if args.metric == 'memo':
        memo = MEMO(
            data_path,
            save_path,
            model_name_or_path,
            tokenizer_path
        )
        
        start_time = time.time()
        memo.run(args.mode)
        end_time = time.time()
        robot.post_message(f'MEMO Finished! Mode:{args.mode}, Time Used: {round(end_time - start_time)}!')
        
    elif args.metric == 'el':
        el = EL(
            data_path,
            save_path,
            model_name_or_path,
            tokenizer_path
        )
        
        start_time = time.time()
        el.run(args.mode)
        end_time = time.time()
        robot.post_message(f'EL Finished! Mode:{args.mode}, Time Used: {round(end_time - start_time)}!')
        
    elif args.metric == 'ma':
        ma = MA(
            data_path,
            save_path,
            model_name_or_path,
            tokenizer_path
        )

        start_time = time.time()
        ma.run(args.mode)
        end_time = time.time()
        robot.post_message(f'MA Finished! Mode:{args.mode}, Time Used: {round(end_time - start_time)}!')
        
    else:
        raise NotImplementedError
        
        