#!/bin/bash -l
#SBATCH --job-name=prep_stv
#SBATCH --partition=shared
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/home/people/25271293/cSTAR_rols/logs/prep_stv_%x_%j.out
#SBATCH --error=/home/people/25271293/cSTAR_rols/logs/prep_stv_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ilaria.benciolini@ucdconnect.ie

CONDITION="${CONDITION:-Rest}"
DONORS="D1_D2_D3_D4"
BASE=~/cSTAR_rols

module load anaconda3/2024.10-1
conda activate rols_env

echo "Node: $(hostname)"
echo "Condition: $CONDITION"
echo "CPUs: $SLURM_CPUS_ON_NODE"

# Rest reuses v_stim from the Stim8hr checkpoint dir -- only needed for Rest,
# harmless to pass unconditionally since script only reads it when required.
VSTIM_REF_ARG=""
if [ "$CONDITION" == "Rest" ]; then
    VSTIM_REF_ARG="--vstim_ref_dir $BASE/checkpoints/Stim8hr"
fi

python $BASE/scripts/1_prep_stv.py \
    --condition $CONDITION \
    --donors $DONORS \
    --data_dir $BASE/Data \
    --ckpt_dir $BASE/checkpoints/$CONDITION \
    $VSTIM_REF_ARG
