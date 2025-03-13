#!/bin/bash
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --gres gpu:H100:3
#SBATCH --output="output32.txt"

# source ../../LLM_attack_env/bin/activate
python RLHF.py
