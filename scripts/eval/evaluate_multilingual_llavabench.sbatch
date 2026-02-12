#!/bin/bash
#SBATCH --job-name=multilingual_llavabench
#SBATCH --account=EUHPC_R04_192
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120

# Arguments
LANGUAGE=${1}           # Language: arabic, bengali, chinese, english, french, hindi, japanese, russian, spanish, urdu
MODEL_PATH=${2}         # Model checkpoint path
OUTPUT_PATH=${3}        # Output JSON path

# ---------------- Env ----------------
source .venv/bin/activate
module load openjdk/11.0.20.1_1
module load ninja/1.11.1

export WANDB_DISABLED=true
export HF_HOME=/leonardo_scratch/large/userexternal/fmohamma/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_HUB_OFFLINE=1

# Data paths
DATA_ROOT=/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/llava_data/eval_benchmarks
MULTILINGUAL_DIR=$DATA_ROOT/multilingual-llava-bench-in-the-wild_reformatted_correct

echo "====================================="
echo "Language: $LANGUAGE"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_PATH"
echo "====================================="

nvidia-smi

# Create output directory if needed
mkdir -p $(dirname "$OUTPUT_PATH")

# Run evaluation
python llava/eval/evaluate_multilingual_captioning.py \
    --model-path "$MODEL_PATH" \
    --multilingual_llavabench_image_path "$MULTILINGUAL_DIR/images" \
    --multilingual_llavabench_questions_path "$MULTILINGUAL_DIR/$LANGUAGE/question.jsonl" \
    --multilingual_llavabench_answers_path "$MULTILINGUAL_DIR/$LANGUAGE/answers.jsonl" \
    --language "$LANGUAGE" \
    --output "$OUTPUT_PATH" \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max-new-tokens 50 \
    --debug

echo "Completed evaluation for $LANGUAGE"
