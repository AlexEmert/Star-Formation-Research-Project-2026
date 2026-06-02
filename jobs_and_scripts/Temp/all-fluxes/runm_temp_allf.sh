#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J temp_model_fit
#SBATCH -p bsudfq
#SBATCH -t 04:00:00
#SBATCH --output=out_temp_allf.out


. ~/.bashrc
mamba activate starform-alex
python mod_temp_allf.py