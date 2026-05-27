#!/usr/bin/env bash
#SBATCH -c 48
#SBATCH -N 1
#SBATCH -J catboost_model_fit
#SBATCH -p bsudfq
#SBATCH -t 00:30:00


. ~/.bashrc
mamba activate starform-alex
python catmodel1.py