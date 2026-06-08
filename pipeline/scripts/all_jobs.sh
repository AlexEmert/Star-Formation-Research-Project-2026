#!/bin/bash

RESPONSES=("LRATIO" "T_BOL") #"LM" "L_BOL" "MASS" "DIAM" "SURF_DENS" "TEMP")
SPACES=("rf_space.pkl") # "catboost_space.pkl" "xgboost_space.pkl" "tree_space.pkl")


# just in case something goes wrong, automatically makes results and logs folders
mkdir -p ../logs ../results


for r in "${RESPONSES[@]}"; do
    for s in "${SPACES[@]}"; do

        sbatch --export=ALL,RESPONSE="$r",SPACE="$s",ITERS="30" run_job.sh

        sleep 0.1
            
    done
done