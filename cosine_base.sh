#!/bin/bash
#
#SBATCH --job-name=triplet_base
#SBATCH --output=logs/triplet_base.txt
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:2
srun python -u main_triplet.py