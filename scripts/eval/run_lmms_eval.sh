#!/bin/bash
# Run lmms-eval evaluation for LLaVA Apertus models.
#
# This script uses the lmms-eval submodule to evaluate LLaVA Apertus (or any
# other LLaVA model) on a configurable set of benchmark tasks.
#
# Usage:
#   bash scripts/eval/run_lmms_eval.sh [MODEL_PATH] [TASKS] [NUM_GPUS] [BATCH_SIZE] [OUTPUT_DIR]
#
# Examples:
#   # Evaluate llava-next-apertus-8b on default image tasks
#   bash scripts/eval/run_lmms_eval.sh /path/to/llava-next-apertus-8b-finetune-full
#
#   # Evaluate on specific tasks with 4 GPUs
#   bash scripts/eval/run_lmms_eval.sh /path/to/model "ai2d,chartqa,mme" 4
#
# Prerequisites:
#   1. Install lmms-eval from the submodule:
#        bash scripts/setup_lmms_eval.sh
#   2. Install the llava package (see README.md)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ---- Configuration (override via arguments or environment) ----
MODEL_PATH="${1:-${LLAVA_MODEL_PATH:-}}"
# Default benchmark tasks for LLaVA Apertus evaluation
DEFAULT_TASKS="ai2d,chartqa,docvqa_val,infovqa_val,mme,realworldqa"
DEFAULT_TASKS="${DEFAULT_TASKS},mathvista_testmini,mmvet,mmbench_en_dev,ocrbench"
DEFAULT_TASKS="${DEFAULT_TASKS},mmmu,seedbench,scienceqa_img,mmstar"

TASKS="${2:-$DEFAULT_TASKS}"
NUM_GPUS="${3:-${LLAVA_NUM_GPUS:-1}}"
BATCH_SIZE="${4:-${LLAVA_BATCH_SIZE:-1}}"
OUTPUT_DIR="${5:-${LLAVA_OUTPUT_DIR:-./logs}}"
CONV_TEMPLATE="${LLAVA_CONV_TEMPLATE:-qwen_1_5}"
MODEL_NAME="${LLAVA_MODEL_NAME:-llava_qwen}"
EXTRA_ARGS="${LLAVA_EXTRA_ARGS:-}"

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is required."
    echo "Usage: $0 <model_path> [tasks] [num_gpus] [batch_size] [output_dir]"
    exit 1
fi

# Verify lmms-eval is available
if ! python -c "import lmms_eval" 2>/dev/null; then
    echo "Error: lmms_eval is not installed."
    echo "Run 'bash scripts/setup_lmms_eval.sh' first to install from the submodule."
    exit 1
fi

echo "==> Running lmms-eval evaluation"
echo "    Model      : $MODEL_PATH"
echo "    Tasks      : $TASKS"
echo "    GPUs       : $NUM_GPUS"
echo "    Batch size : $BATCH_SIZE"
echo "    Output dir : $OUTPUT_DIR"

accelerate launch --num_processes="$NUM_GPUS" \
    -m lmms_eval \
    --model llava_onevision \
    --model_args "pretrained=$MODEL_PATH,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --log_samples_suffix llava_apertus \
    --output_path "$OUTPUT_DIR" \
    $EXTRA_ARGS
