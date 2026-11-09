#!/bin/bash -l
#$ -P ece601
#$ -N cage
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=12:00:00
#$ -j y
#$ -o /projectnb/ece601/colemian/runs/logs/$JOB_NAME.$JOB_ID.log

cd /projectnb/ece601/colemian/EC601
source env/scc.sh
python scripts/check_env.py | head -3      # records the environment at the top of every log
python "$@"