#!/bin/bash
# Job name:
#SBATCH --job-name=Etrap_vertical_v1
#
# Account:
#SBATCH --account=fc_haeffnerbem
#
# Partition:
#SBATCH --partition=savio3
#
# Request one node:
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=1:00:00
#
## Command(s) to run:
module load python/3.9.12
source activate bem39
ipython run.py > job.pyout
