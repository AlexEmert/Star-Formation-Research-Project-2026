#!/usr/bin/env bash
#SBATCH -c 48
#SBATCH -N 1
#SBATCH -J rf_model_fit
#SBATCH -p bsudfq
#SBATCH -t 00:30:00
#SBATCH --output=rf_all_fluxes_results.out


. ~/.bashrc
mamba activate starform-alex
python rf_all_fluxes.py