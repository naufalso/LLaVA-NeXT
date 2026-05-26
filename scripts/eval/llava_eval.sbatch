#!/bin/bash
#SBATCH --job-name=llava_eval_dbg
#SBATCH --account=EUHPC_R04_192
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # one launcher per node
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/out/%x-%j.out
#SBATCH --error=./logs/err/%x-%j.out
#SBATCH --signal=B:TERM@120           # give 120s to checkpoint on preemption/node issues

# --qos=boost_qos_dbg

# ---------------- Env ----------------
source .venv/bin/activate
module load openjdk/11.0.20.1_1
module load ninja/1.11.1

export WANDB_DISABLED=true

export WANDB_DISABLED=true
# export HF_HOME=/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/llava_data/weights/.cache/huggingface
export HF_HOME=/leonardo_scratch/large/userexternal/fmohamma/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_HUB_OFFLINE=1    # enforce offline mode

DATA_ROOT=${1:-/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/llava_data/eval_benchmarks}
MODEL_PATH=${2:-liuhaotian/llava-v1.5-7b}
load_encoder=${3:-none}
DEVICE=${4:-2}
flickr=${5:-false}
coco=${6:-false}
vqav2=${7:-false}
textvqa=${8:-false}
vizwiz=${9:-false}
okvqa=${10:-false}
mt_trust=${11:-false}
eval_multilingual_llavabench=${12:-false}
eval_multilingual_llavabench_lang=${13:-english}
# arabic  bengali  chinese  english  french  hindi  images  japanese  russian  spanish  urdu
attack=${14:-none}
eps=${15:-2}
steps=${16:-100}
results_folder=${17:-Evaluations_Leonardo}

# echo all arguments
echo "All arguments: $@"

nvidia-smi

# LLaVA evaluation script
#CUDA_VISIBLE_DEVICES=$DEVICE
python -m vlm_eval.run_evaluation \
--load_encoder $load_encoder \
--results_folder $results_folder \
--eval_flickr30 $flickr \
--eval_coco $coco \
--eval_vqav2 $vqav2 \
--eval_textvqa $textvqa \
--eval_vizwiz $vizwiz \
--eval_ok_vqa $okvqa \
--eval_multilingual_llavabench $eval_multilingual_llavabench \
--eval_flickr30_lang english \
--eval_coco_lang english \
--eval_vqav2_lang english \
--eval_textvqa_lang english \
--eval_vizwiz_lang english \
--eval_ok_vqa_lang english \
--eval_multitrust $mt_trust \
--attack $attack --eps $eps --steps $steps --mask_out none \
--vision_encoder_pretrained none \
--precision float16 \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.8 \
--num_beams 1 \
--out_save_path $MODEL_PATH \
--out_base_path ./results_debug \
--model_path $MODEL_PATH \
--coco_train_image_dir_path $DATA_ROOT/coco/train2014 \
--coco_val_image_dir_path $DATA_ROOT/coco/val2014 \
--coco_karpathy_json_path $DATA_ROOT/coco/annotations/karpathy_coco.json \
--coco_annotations_json_path $DATA_ROOT/coco/annotations/captions_val2014.json \
--flickr_image_dir_path $DATA_ROOT/flickr30k/flickr30k-images \
--flickr_karpathy_json_path $DATA_ROOT/flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path $DATA_ROOT/flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path $DATA_ROOT/vizwiz/train \
--vizwiz_test_image_dir_path $DATA_ROOT/vizwiz/val \
--vizwiz_train_questions_json_path $DATA_ROOT/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path $DATA_ROOT/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path $DATA_ROOT/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path $DATA_ROOT/vizwiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path $DATA_ROOT/coco/train2014 \
--vqav2_train_questions_json_path $DATA_ROOT/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path $DATA_ROOT/vqav2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path $DATA_ROOT/coco/val2014 \
--vqav2_test_questions_json_path $DATA_ROOT/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path $DATA_ROOT/vqav2/v2_mscoco_val2014_annotations.json \
--ok_vqa_train_image_dir_path $DATA_ROOT/coco/train2014 \
--ok_vqa_train_questions_json_path $DATA_ROOT/okvqa/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path $DATA_ROOT/okvqa/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path $DATA_ROOT/coco/val2014 \
--ok_vqa_test_questions_json_path $DATA_ROOT/okvqa/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path $DATA_ROOT/okvqa/mscoco_val2014_annotations.json \
--textvqa_image_dir_path $DATA_ROOT/textvqa/train_images \
--textvqa_train_questions_json_path $DATA_ROOT/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path $DATA_ROOT/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path $DATA_ROOT/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path $DATA_ROOT/textvqa/val_annotations_vqa_format.json \
--mt_data_path $DATA_ROOT/MultiTrust/robustness/adv_nips \
--multilingual_llavabench_image_path  $DATA_ROOT/multilingual-llava-bench-in-the-wild_reformatted_correct/images \
--multilingual_llavabench_questions_path $DATA_ROOT/multilingual-llava-bench-in-the-wild_reformatted_correct/$eval_multilingual_llavabench_lang/question.jsonl \
--multilingual_llavabench_answers_path $DATA_ROOT/multilingual-llava-bench-in-the-wild_reformatted_correct/$eval_multilingual_llavabench_lang/answers.jsonl \
--verbose