#!/usr/bin/env bash
#SBATCH -c 48
#SBATCH -N 1
#SBATCH -J temp_model_fit
#SBATCH -p bsudfq
#SBATCH -t 12:00:00


. ~/.bashrc
mamba activate starform-alex
python temp_best_model_low_fluxes.py
