#!/usr/bin/env bash
#SBATCH -c 48
#SBATCH -N 1
#SBATCH -J cat_model_fit_all_fluxes
#SBATCH -p bsudfq
#SBATCH -t 01:00:00
#SBATCH --output=catboost-all_fluxes_results.out

. ~/.bashrc
mamba activate starform-alex
python catmodel_all_fluxes.py