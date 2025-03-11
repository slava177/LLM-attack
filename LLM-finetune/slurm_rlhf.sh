#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --gres gpu:H100:2
#SBATCH --output="output14.txt"

# source ../../LLM_attack_env/bin/activate
python RLHF.py
