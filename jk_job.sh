#!/bin/bash	

#SBATCH --job-name=jk_model
#SBATCH --output=jk_model_%J.out
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:Turing:1
#SBATCH --mail-user=s214704@dtu.dk
#SBATCH --mail-type=BEGIN,END,FAIL
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
source bsc-env/bin/activate
python VertebraeSegmentation/Verse/Training.py --no-mps 

echo "Done: $(date +%F-%R:%S)"