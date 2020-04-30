#!/bin/bash
# Run GAE model

#SBATCH --constraint=v100
#SBATCH --partition gpu
#SBATCH --gres=gpu:01
#SBATCH --job-name mtae

cd $HOME/projects/ProteinUniverse
source $HOME/anaconda3/etc/profile.d/conda.sh
source $HOME/.node_allocation/load_gpu_modules

conda activate protuniv

#LEVEL=C
#LEVEL=A
LEVEL=T

FILTERS="64 64 64 64 64"
#FILTERS="64 128 128 256"
#FILTERS="128 128 128 128"



#THRESHOLD=4
THRESHOLD=6
#THRESHOLD=8
#THRESHOLD=10

ARCH_SESSION=$(echo "${FILTERS}" | sed "s/\s/\-/g")
SESSION=${ARCH_SESSION}__L${LEVEL}__T${THRESHOLD}


python train_multitsk.py \
    --filter-dims $FILTERS \
    --l2-reg 5e-6 \
    --lr 0.001 \
    --epochs 30 \
    --batch-size 64 \
    --level $LEVEL \
    --threshold $THRESHOLD \
    --results_dir ./results/multitask/max/$LEVEL/$SESSION 
