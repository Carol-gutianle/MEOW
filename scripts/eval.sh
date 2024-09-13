model_name="MODEL PATH"
data_set="forget10"
cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")
sbatch -p AI4GOV --gres=gpu:1 --quotatype=reserved --wrap="python -m meow.tofu.eval model_path=$model_name method_name=$date_dir-$cur_time data_set=$data_set"