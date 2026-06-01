#!/bin/bash
#SBATCH --job-name="dst_llms"
#SBATCH --account="a-***"
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --partition=normal            # debug; normal; according to the partition you want to use
#SBATCH --environment=my_env

#srun --environment=my_env bash << 'EOF'

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"
echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345 # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "[sbatch-master] execute command on compute nodes"

export NORM_TYPE="pre"
export POST_NUM=3

export size=$3

# Set density based on model size
export density=1.0

export epochs=$1
export training_steps=$2
export batch=$5

export model_name="llama"
export seed=0
export learning_rate=$4

export optimizer="adam"
export weight_decay=0.0
export growth="gradient"      # SET: "random", RigL: "gradient"
export prune="magnitude"  # "magnitude_soft" or "magnitude"
export temperature=3.0
export prune_rate=0.1
export update_freq=100

export acc_grad_steps=5  ## gradient_acc
export maintain_times=2  ## gradient_acc
export reinit='zero'

export fix=True
export prune_rate_decay="cosine"  # constant, cosine, WSD
export am_ratio=1.0
export sparse_init="uniform"  # fixed_ERK; uniform; uniform_ratio

export warmup_steps=1000

export run_name="${size}_s${seed}"

export max_length=256
export total_batch_size=$((batch*4))

export val_dir="./data/llms/c4_sampling/c4_filtered_validation_10M"
export data_dir="./data/llms/c4_sampling/c4_filtered_maxlength${max_length}_bs512_step${training_steps}_arrow_shuffle32"
#data_dir=None

export output_dir="./$USER/logs/dst_llm/mul_nodes_dense/model_${model_name}${size}_c4_f_l${max_length}_bs${total_batch_size}_step${training_steps}_g${growth}${acc_grad_steps}${maintain_times}_reinit${reinit}_p${prune}${prune_rate}_${prune_rate_decay}_f${update_freq}_d${density}_init${sparse_init}_am${am_ratio}_fix${fix}_wp${warmup_steps}_lr${learning_rate}_ep${epochs}_steps${training_steps}_op${optimizer}wd${weight_decay}_nodes${SLURM_NNODES}_exfl"

mkdir -p "${output_dir}/checkpoints"

export log_file="${output_dir}/log1.txt"

CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

echo \" \${data_dir} \${run_name}\"

torchrun \
    --nnodes="${SLURM_NNODES}" \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ./torchrun_main.py \
    --wandb_mode disabled \
    --seed "\${seed}" \
    --model_config "./configs_new/llama_\${size}.json" \
    --density "\${density}" \
    --val_dir "\${val_dir}" \
    --data_dir "\${data_dir}" \
    --update_frequency "\${update_freq}" \
    --growth "\${growth}" \
    --accumulate_grad_steps "\${acc_grad_steps}" \
    --maintain_num "\${maintain_times}" \
    --reinit "\${reinit}" \
    --prune "\${prune}" \
    --prune_rate "\${prune_rate}" \
    --prune_rate_decay "\${prune_rate_decay}" \
    --temperature "\${temperature}" \
    --sparse_init "\${sparse_init}" \
    --am_ratio "\${am_ratio}" \
    --fix "\${fix}" \
    --lr "\${learning_rate}" \
    --optimizer "\${optimizer}" \
    --weight_decay "\${weight_decay}" \
    --batch_size "\${batch}" \
    --total_batch_size "\${total_batch_size}" \
    --num_training_steps "\${training_steps}" \
    --epochs "\${epochs}" \
    --warmup_steps "\${warmup_steps}" \
    --eval_every 2000 \
    --dtype bfloat16 \
    --grad_clipping 0.0 \
    --run_name "\${run_name}" \
    --save_dir "\${output_dir}/checkpoints"

"
#srun bash -c "
srun bash -c "$CMD" > "${log_file}" 2>&1

#EOF
