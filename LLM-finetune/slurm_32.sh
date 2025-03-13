#!/bin/bash
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --gres gpu:H100:1
#SBATCH --output="output7.txt"

# source ../../LLM_attack_env/bin/activate
python test_response7.py
