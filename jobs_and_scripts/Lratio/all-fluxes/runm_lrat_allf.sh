#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J lrat_model_fit
#SBATCH -p bsudfq
#SBATCH -t 1-00:00:00
#SBATCH --output=out_lrat_allf.out


. ~/.bashrc
mamba activate starform-alex
python mod_lrat_allf.py