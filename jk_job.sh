#!/bin/bash	

#SBATCH --job-name=jk_model
#SBATCH --output=jk_model_%J.out
#SBATCH --cpus-per-task=6
#SBATCH --time=4-12:00:00
#SBATCH --mem=12gb
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

source "C:\Users\julie\.conda"
source ~/.bashrc
module load CUDA/11.4
conda activate bscenv
python VertebraeSegmentation/Verse/Training.py --no-mps 

echo "Done: $(date +%F-%R:%S)"