#!/bin/bash

JOB_ID=27531941
JOB_ID=$(sbatch -d afterany:$JOB_ID /leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/scripts/train/run_llava_apertus_finetune_full_lora.sbatch | awk '{print $4}')
JOB_ID=$(sbatch -d afterany:$JOB_ID /leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/scripts/train/run_llava_apertus_finetune_full_lora.sbatch | awk '{print $4}')