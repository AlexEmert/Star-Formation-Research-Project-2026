#!/usr/bin/env bash
#SBATCH -c 48
#SBATCH -t 12:00
#SBATCH -N 1

. ~/.bashrc
mamba activate starform-alex
python fit_models.pyls
