#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J lm_model_fit
#SBATCH -p bsudfq
#SBATCH -t 03:00:00
#SBATCH --output=out_lm_lowf.out


. ~/.bashrc
mamba activate starform-alex
python mod_lm_lowf.py