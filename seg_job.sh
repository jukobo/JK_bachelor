#!/bin/bash	

#SBATCH --job-name=jk_prep_stage2_spine_model
#SBATCH --output=jk_prep_stage2_spine_model_%J.out
#SBATCH --cpus-per-task=2
#SBATCH --time=1-05:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:2
#SBATCH --mail-user=s214725@dtu.dk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

source ~/JK_bachelor/.bashrc
module load CUDA/11.4
python VertebraeLocalisation_singleclass/Verse/Data_preprocessing/Preprocessing_NOPADDING.py --no-mps 



echo "Done: $(date +%F-%R:%S)"