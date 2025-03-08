#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5:00:00;
#SBATCH --gres gpu:H200:1
#SBATCH --output="output32.txt"

# source ../../LLM_attack_env/bin/activate
python new_lora.py
