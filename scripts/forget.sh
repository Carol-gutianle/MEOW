set -x

PARTITION=${PARTITION:-"AI4GOV"}
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")
OUTPUT_DIR=$date_dir

random_number=$((RANDOM % 10001 + 20000))
max_length=200
top_k=3
metric=largest
max_steps=-1
forget_loss=grad_ascent_KL

meow_args="++max_length=$max_length ++top_k=$top_k ++metric=$metric ++max_steps=$max_steps ++forget_loss=$forget_loss"

mkdir -p $OUTPUT_DIR/${cur_time}
# forget
sbatch -p ${PARTITION} \
  -o $OUTPUT_DIR/${cur_time}/length_${max_length}-top_${top_k}-metric_${metric}.out \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${random_number} -m meow.tofu.forget $meow_args \