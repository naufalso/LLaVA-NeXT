#!/bin/bash
#SBATCH --job-name=llava_apertus_inference_dbg
#SBATCH --account=EUHPC_R04_192
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # one launcher per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/out/%x-%j.out
#SBATCH --error=./logs/err/%x-%j.out
#SBATCH --signal=B:TERM@120           # give 120s to checkpoint on preemption/node issues
#SBATCH --qos=boost_qos_dbg


#--qos=boost_qos_dbg

export WANDB_DISABLED=true
# export HF_HOME=/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/llava_data/weights/.cache/huggingface
export HF_HOME=/leonardo_scratch/large/userexternal/fmohamma/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_HUB_OFFLINE=1    # enforce offline mode

# NCCL (set to your fabric; commonly ib0 on Leonardo)
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "--- Slurm Job Setup ---"
echo "Job ID: $SLURM_JOB_ID"
echo "----------------- Slurm Job Setup -----------------"
echo " Job ID:               $SLURM_JOB_ID"
echo " Job Name:             $SLURM_JOB_NAME"
echo " Partition:            $SLURM_JOB_PARTITION"
echo " Account:              $SLURM_JOB_ACCOUNT"
echo " Submit Host:          $SLURM_SUBMIT_HOST"
echo " Submit Dir:           $SLURM_SUBMIT_DIR"
echo " Nodes allocated:      $SLURM_JOB_NUM_NODES"
echo " Tasks per node:       $SLURM_NTASKS_PER_NODE"
echo " CPUs per task:        $SLURM_CPUS_PER_TASK"
echo " GPUs per node:        $SLURM_GPUS_ON_NODE"
echo " Node list:            $SLURM_JOB_NODELIST"
scontrol show hostnames "$SLURM_NODELIST"
echo
echo " Rendezvous Setup:"
# ---------- Torch-distributed rendezvous ----------
NUM_NODES=${SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}   
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=$((20000 + RANDOM % 10000))
export MASTER_ADDR MASTER_PORT
RDZV_ID=${SLURM_JOB_ID}
echo "   MASTER_ADDR:        $MASTER_ADDR"
echo "   MASTER_PORT:        $MASTER_PORT"
echo "   RDZV_ID:            $RDZV_ID"
echo "---------------------------------------------------"

echo "Launching with torchrun across $SLURM_JOB_NUM_NODES nodes x $SLURM_GPUS_ON_NODE GPUs"

source .venv/bin/activate

nvidia-smi

# llava/serve/examples/extreme_ironing.jpg

echo "Starting inference test..."
python -m llava.serve.cli \
    --model-path /leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/checkpoints/llava-next-apertus-8b-finetune-full \
    --image-file /leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/playground/eval_data/flickr30k/flickr30k-images/4464247181.jpg \
    --non-interactive \
    --temperature 0.2 \
    --debug
echo "Inference test completed."
