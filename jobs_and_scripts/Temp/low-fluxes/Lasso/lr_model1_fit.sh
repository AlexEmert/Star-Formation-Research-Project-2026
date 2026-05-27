#!/usr/bin/env bash
#SBATCH -c 48
#SBATCH -N 1
#SBATCH -J linear_model_fit
#SBATCH -p bsudfq
#SBATCH -t 00:10:00


. ~/.bashrc
mamba activate starform-alex
python lrmodel1.py