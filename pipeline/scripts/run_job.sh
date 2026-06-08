#!/bin/bash
#SBATCH -J model_fit      # job name
#SBATCH -n 1              # total number of tasks requested
#SBATCH -c 48             # CPU cores per task
#SBATCH -N 1              # number of nodes you want to run on
#SBATCH -p bsudfq         # queue (partition)
#SBATCH -t 1-00:00:00       # run time (hh:mm:ss)
#SBATCH --output=../logs/%x_%j.out # output and error file name (%j expands to jobID)


. ~/.bashrc
mamba activate starform-alex

python search.py \
    --response "$RESPONSE" \
    --space "$SPACE" \
    --iters "$ITERS"