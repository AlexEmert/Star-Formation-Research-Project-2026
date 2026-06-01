#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -J svr_poly_model_fit
#SBATCH -p bsudfq
#SBATCH -t 06:00:00
#SBATCH --output=svr_poly_test_results.out


. ~/.bashrc
mamba activate starform-alex
python svr_poly.py