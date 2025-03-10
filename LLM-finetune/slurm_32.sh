#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --gres gpu:H100:1
#SBATCH --output="output14.txt"

# source ../../LLM_attack_env/bin/activate
python test_response2.py
