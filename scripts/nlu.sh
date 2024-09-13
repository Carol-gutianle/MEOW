model_name_or_path="{MODEL PATH}"
tokenizer_path="{TOKENIZER PATH}"
sbatch -p AI4GOV --gres=gpu:1 --wrap="python -m evaluator.nlu --split all --model_name_or_path $model_name_or_path --tokenizer_path $tokenizer_path"