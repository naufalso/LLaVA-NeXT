#!/bin/bash

current_date=$(date +"%Y%m%d")
log_dir="logs/eval/$current_date"
mkdir -p "$log_dir"

EVAL_SCRIPT="scripts/eval/evaluate_multilingual_llavabench.sbatch"

# All supported languages
languages=(arabic bengali chinese english french hindi japanese russian spanish urdu)

# Models to evaluate
models=(
	"checkpoints/llava-next-apertus-siglip2-8b-finetune-full-ori-fixed|llava-v1.5-apertus-siglip2-8b-fixed"
	# "checkpoints/llava-next-apertus-8b-finetune-full-apertusprompt|llava-v1.5-apertus-8b"
	# "checkpoints/llama3-llava-next-8b|official-llava-next-llama3-8b"
	# "checkpoints/llava-v1.5-7b|official-llava-v1.5-7b"
	# "checkpoints/llava-next-apertus-8b-finetune-full-ori-fixed|llava-v1.5-apertus-8b-fixed"
	# "checkpoints/llava-next-apertus-siglip-8b-finetune-full-ori-fixed|llava-v1.5-apertus-siglip-8b-fixed"
	# "checkpoints/llava-v1.5-finetune-full-v1|llava-v1.5"
	# "checkpoints/llava-next-llama3-8b-finetune-full-llama3prompt|llava-next-llama3-8b"
	# "checkpoints/llava-next-qwen25-7b-finetune-full-qwen25prompt|llava-next-qwen25-7b"
)

for language in "${languages[@]}"; do
	for entry in "${models[@]}"; do
		checkpoint="${entry%%|*}"
		stem="${entry##*|}"
		output="./results_debug/eval_multilingual_llavabench/${language}/${stem}-${language}.json"
		
		echo "Submitting: $stem - $language"
		sbatch --output=$log_dir/%x-%j.out --error=$log_dir/%x-%j.err --job-name=multilingual_llavabench_${stem}_${language} \
			$EVAL_SCRIPT "$language" "$checkpoint" "$output"
	done
done

echo "All jobs submitted. Check logs in $log_dir"
