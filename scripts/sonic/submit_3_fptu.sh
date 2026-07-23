#!/bin/bash -l
#SBATCH --job-name=fptu
#SBATCH --partition=shared
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/home/people/25271293/cSTAR_rols/logs/fptu_%x_%j.out
#SBATCH --error=/home/people/25271293/cSTAR_rols/logs/fptu_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ilaria.benciolini@ucdconnect.ie

CONDITION="${CONDITION:-Rest}"
DONORS="D1_D2_D3_D4"
METHOD="${METHOD:-ols}"
BASE=~/cSTAR_rols

module load anaconda3/2024.10-1
conda activate rols_env

echo "Node: $(hostname)"
echo "Condition: $CONDITION"
echo "Method: $METHOD"

python $BASE/scripts/3_fptu.py \
    --condition $CONDITION \
    --donors $DONORS \
    --method $METHOD \
    --ckpt_dir $BASE/checkpoints/$CONDITION
