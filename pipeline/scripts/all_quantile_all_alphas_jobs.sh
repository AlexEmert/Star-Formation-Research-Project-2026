#!/bin/bash

RESPONSES=("LRATIO" "LM" "L_BOL" "MASS" "DIAM" "SURF_DENS" "TEMP" "T_BOL")
THRESHOLDS=("0.5" "1.5")

# just in case something goes wrong, automatically makes results and logs folders
mkdir -p ../logs ../results


for r in "${RESPONSES[@]}"; do
   for t in "${THRESHOLDS[@]}"; do

      sbatch --export=ALL,RESPONSE="$r",ITERS="320",THRESHOLD="$t" run_residual_model.sh

      sleep 0.1

   done
done