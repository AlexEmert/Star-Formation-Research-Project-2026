#!/bin/bash
#SBATCH -J model_fit      # job name
#SBATCH -n 1              # total number of tasks requested
#SBATCH -c 48             # CPU cores per task
#SBATCH -N 1              # number of nodes you want to run on
#SBATCH -p bsudfq         # queue (partition)
#SBATCH -t 2-00:00:00     # run time (hh:mm:ss)
#SBATCH --output=../logs/%x_%j.out # output and error file name (%j expands to jobID)


. ~/.bashrc
mamba activate starform-alex

# python optuna_search_mult_quant_reg.py \
#     --iters 10 \
#     --response "TEMP" 

python optuna_search_residual_models.py \
    --iters 10 \
    --response "TEMP"  

# python optuna_search_mult_quant_reg.py \
#     --iters "$ITERS" \
#     --response "$RESPONSE" \
#     --threshold "$THRESHOLD" 
