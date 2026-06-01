#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J log_temp_model_fit
#SBATCH -p bsudfq
#SBATCH -t 00:30:00
#SBATCH --output=log_temp_model_test.out


. ~/.bashrc
mamba activate starform-alex
python logging-temp.py
