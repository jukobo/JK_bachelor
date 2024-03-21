#!/bin/bash	

#SBATCH --job-name=jk_test
#SBATCH --output=jk_test_%J.out
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:Turing:1
#SBATCH --mail-user=s214704@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

source ~/.bashrc
module load CUDA/11.4
source bscenv/bin/activate
python J_test.py --no-mps 

echo "Done: $(date +%F-%R:%S)"