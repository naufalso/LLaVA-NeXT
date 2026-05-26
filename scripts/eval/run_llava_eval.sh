#!/bin/bash

MODEL_PATH=${1:-/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/checkpoints/llava-next-apertus-8b-finetune-full}
DATA_ROOT=${2:-/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/playground/eval_data}
ENCODER=${3:-none}
ATTACK=${4:-none}
EPSILON=${5:-2}
STEPS=${6:-100}
TASK=${7:-0}
GPU=0

# flickr=${5:-false}
# coco=${6:-false}
# vqav2=${7:-false}
# textvqa=${8:-false}
# vizwiz=${9:-false}
# okvqa=${10:-false}
# mt_trust=${11:-false}

if [ $TASK -eq 0 ]; then

#   echo "All task will be evaluated"

    # Evaluate flickr
    sbatch --job-name=llava_eval_flickr_verbose scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false false false english $ATTACK $EPSILON $STEPS

    # sbatch --job-name=llava_eval_coco_verbose scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false false false english $ATTACK $EPSILON $STEPS

    # sbatch scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false false false english $ATTACK $EPSILON $STEPS

    # sbatch scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false false false english $ATTACK $EPSILON $STEPS

    # sbatch scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false false false english $ATTACK $EPSILON $STEPS

    # sbatch scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true false false english $ATTACK $EPSILON $STEPS

    # sbatch scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false true false english $ATTACK $EPSILON $STEPS
    
    # sbatch scripts/eval/llava_eval.sbatch $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true english $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true arabic $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true bengali $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true chinese $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true french $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true hindi $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true japanese $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true russian $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true spanish $ATTACK $EPSILON $STEPS

#   bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false false true urdu $ATTACK $EPSILON $STEPS

fi



#
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false ensemble 4
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false ensemble 4
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false ensemble 4
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false ensemble 4
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false ensemble 4
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true ensemble 4
#
#
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false ensemble 8
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false ensemble 8
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false ensemble 8
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false ensemble 8
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false ensemble 8
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true ensemble 8

# bash script.sh model_path load_encoder GPU flickr coco vqav2 textvqa vizwiz okvqa multitrust attack eps

#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false false none $EPSILON
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false false none $EPSILON
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false false none $EPSILON
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false false none $EPSILON
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false false none $EPSILON
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true false none $EPSILON
#
#bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false true none $EPSILON
#
#
#bash bash/eval_pope.sh $DATA_ROOT  $MODEL_PATH $ENCODER
#
#bash bash/eval_scienceqa.sh $DATA_ROOT  $MODEL_PATH $ENCODER