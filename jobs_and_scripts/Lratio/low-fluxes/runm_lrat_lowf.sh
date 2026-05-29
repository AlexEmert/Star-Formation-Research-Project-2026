#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J lrat_model_fit
#SBATCH -p bsudfq
#SBATCH -t 03:00:00
#SBATCH --output=out_lrat_lowf.out


. ~/.bashrc
mamba activate starform-alex
python mod_lrat_lowf.py