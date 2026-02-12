#!/bin/bash
#SBATCH --job-name=llava_vqa_eval
#SBATCH --account=EUHPC_R04_192
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # one launcher per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/out/%x-%j.out
#SBATCH --error=./logs/err/%x-%j.out
#SBATCH --signal=B:TERM@120           # give 120s to checkpoint on preemption/node issues


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
module load openjdk/11.0.20.1_1
module load ninja/1.11.1

nvidia-smi

echo "Starting VQA eval..."
DATASET=${1:-vqav2}
MODEL_PATH=${2:-checkpoints/llava-next-apertus-8b-finetune-full-apertusprompt}
OUTPUT=${3:-results_debug/vqav2/llava-next-apertus-8b-vqav2.json}
DATA_ROOT=${4:-/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/llava_data/eval_benchmarks}

if [[ "$DATASET" == "vqav2" ]]; then
	python -m llava.eval.evaluate_vqa \
		--dataset vqav2 \
		--model-path "$MODEL_PATH" \
		--output "$OUTPUT" \
		--temperature 0.0 \
		--max-new-tokens 16 \
		--debug \
        --sample-eval \
		--vqav2-image-dir $DATA_ROOT/coco/val2014 \
		--vqav2-questions-json $DATA_ROOT/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
		--vqav2-annotations-json $DATA_ROOT/vqav2/v2_mscoco_val2014_annotations.json
elif [[ "$DATASET" == "vizwiz" ]]; then
	python -m llava.eval.evaluate_vqa \
		--dataset vizwiz \
		--model-path "$MODEL_PATH" \
		--output "$OUTPUT" \
		--temperature 0.0 \
		--max-new-tokens 16 \
		--debug \
        --sample-eval \
		--vizwiz-image-dir $DATA_ROOT/vizwiz/val \
		--vizwiz-questions-json $DATA_ROOT/vizwiz/val_questions_vqa_format.json \
		--vizwiz-annotations-json $DATA_ROOT/vizwiz/val_annotations_vqa_format.json
elif [[ "$DATASET" == "textvqa" ]]; then
	python -m llava.eval.evaluate_vqa \
		--dataset textvqa \
		--model-path "$MODEL_PATH" \
		--output "$OUTPUT" \
		--temperature 0.0 \
		--max-new-tokens 16 \
		--debug \
        --sample-eval \
		--textvqa-image-dir $DATA_ROOT/textvqa/train_images \
		--textvqa-questions-json $DATA_ROOT/textvqa/val_questions_vqa_format.json \
		--textvqa-annotations-json $DATA_ROOT/textvqa/val_annotations_vqa_format.json
elif [[ "$DATASET" == "okvqa" ]]; then
	python -m llava.eval.evaluate_vqa \
		--dataset ok_vqa \
		--model-path "$MODEL_PATH" \
		--output "$OUTPUT" \
		--temperature 0.0 \
		--max-new-tokens 16 \
		--debug \
        --sample-eval \
		--okvqa-image-dir $DATA_ROOT/coco/val2014 \
		--okvqa-questions-json $DATA_ROOT/okvqa/OpenEnded_mscoco_val2014_questions.json \
		--okvqa-annotations-json $DATA_ROOT/okvqa/mscoco_val2014_annotations.json
else
	echo "Unknown dataset: $DATASET (expected 'vqav2', 'vizwiz', 'textvqa', or 'okvqa')"
	exit 1
fi

echo "VQA eval completed."

#     --sample-size 500 \
