#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J tbol_model_fit
#SBATCH -p bsudfq
#SBATCH -t 1-00:00:00
#SBATCH --output=out_tbol_allf2.out


. ~/.bashrc
mamba activate starform-alex
python mod_tbol_allf.py