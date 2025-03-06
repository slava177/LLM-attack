#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 5:00:00
#SBATCH --gpus=H100:1
#SBATCH --output = "./output.txt"

#source LLM_attack_env/bin/activate
python LLM-attack/LLM-finetune/lora.py