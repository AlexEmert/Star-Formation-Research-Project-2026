#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J tbol_model_fit
#SBATCH -p bsudfq
#SBATCH -t 06:00:00
#SBATCH --output=out_tbol_lowf.out


. ~/.bashrc
mamba activate starform-alex
python mod_tbol_lowf.py