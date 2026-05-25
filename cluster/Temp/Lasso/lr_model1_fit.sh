#!/usr/bin/env bash
#SBATCH -c 48
#SBATCH -N 1

mamba activate starform-alex
python lrmodel1.py
mamba deactivate starform-alex