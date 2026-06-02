#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J dens_model_fit
#SBATCH -p bsudfq
#SBATCH -t 06:00:00
#SBATCH --output=out_dens_allf.out


. ~/.bashrc
mamba activate starform-alex
python mod_dens_allf.py