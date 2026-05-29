#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J temp_model_fit
#SBATCH -p bsudfq
#SBATCH -t 03:00:00
#SBATCH --output=trial3_temp_low_fluxes_results.out


. ~/.bashrc
mamba activate starform-alex
python temp_best_model_low_fluxes.py
