#!/bin/bash

RESPONSES=("LRATIO" "LM" "L_BOL" "MASS" "DIAM" "SURF_DENS" "TEMP" "T_BOL")

# just in case something goes wrong, automatically makes results and logs folders
mkdir -p ../logs ../results


for r in "${RESPONSES[@]}"; do
    sbatch --export=ALL,RESPONSE="$r",ITERS="200" run_job.sh

    sleep 0.1
done