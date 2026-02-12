#!/bin/bash
#SBATCH --job-name=llava_next_apertus_clip
#SBATCH --account=EUHPC_R04_192
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # one launcher per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/out/%x-%j.out
#SBATCH --error=./logs/err/%x-%j.out
#SBATCH --signal=B:TERM@120           # give 120s to checkpoint on preemption/node issues


# ---------------- Env ----------------

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

export WANDB_MODE=offline


LLM_VERSION="playground/models/Apertus-8B-Instruct-2509"
LLM_VERSION_CLEAN="apertus-8B-Instruct-2509"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=apertus

DATA_PATH="./playground/data/blip_laion_cc_sbu_558k.json"
IMAGE_PATH="./playground/data/images"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain_batch256"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

source .venv/bin/activate

nvidia-smi

torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NUM_NODES}" --rdzv_id ${RDZV_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_PATH} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts "mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --attn_implementation sdpa

#     --report_to wandb \
# --run_name $BASE_RUN_NAME \
    # --evaluation_strategy "no" \
