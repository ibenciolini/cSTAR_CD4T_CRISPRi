#!/bin/bash -l
#SBATCH --job-name=rols
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=0
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=/home/people/25271293/cSTAR_rols/logs/rols_%x_%j.out
#SBATCH --error=/home/people/25271293/cSTAR_rols/logs/rols_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ilaria.benciolini@ucdconnect.ie

CONDITION="${CONDITION:-Rest}"
DONORS="D1_D2_D3_D4"
BASE=~/cSTAR_rols

module load anaconda3/2024.10-1
module load cuda/12.9
conda activate rols_env

echo "Node: $(hostname)"
echo "Condition: $CONDITION"

python $BASE/scripts/2_rols.py \
    --condition $CONDITION \
    --donors $DONORS \
    --ckpt_dir $BASE/checkpoints/$CONDITION
