#!/bin/bash

RESPONSES=("LRATIO" "LM" "L_BOL" "MASS" "DIAM" "SURF_DENS" "TEMP" "T_BOL")
THRESHOLDS=("1.5" "2" "2.5" "5" "10")

# just in case something goes wrong, automatically makes results and logs folders
mkdir -p ../logs ../results


#for r in "${RESPONSES[@]}"; do
   # sbatch --export=ALL,RESPONSE="$r",ITERS="80" run_job.sh

   # sleep 0.1
#done

for t in "${THRESHOLDS[@]}"; do
    sbatch --export=ALL,THRESHOLD="$t" run_job.sh

    sleep 0.1
done