#!/bin/bash

RESPONSES=("LRATIO" "LM" "L_BOL" "MASS" "DIAM" "SURF_DENS" "TEMP" "T_BOL")
SPACES=("catboost_space")

# just in case something goes wrong, automatically makes results and logs folders
mkdir -p ../logs ../results


for r in "${RESPONSES[@]}"; do
    for s in "${SPACES[@]}"; do

        sbatch --export=ALL,RESPONSE="$r",SPACE="$s",ITERS="200" run_graph_skopt.sh

        sleep 0.1

    done
done


