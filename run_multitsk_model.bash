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
#LEVEL=T
#LEVEL=H
LEVEL=0

#FILTERS="64 128 128 256"
FILTERS="128 128 128 128"

#POOLING="max"
POOLING="sum"

THRESHOLD=6
ARCH_SESSION=$(echo "${FILTERS}" | sed "s/\s/\-/g")

SESSION=${ARCH_SESSION}__L${LEVEL}__T${THRESHOLD}__P${POOLING}
OUTDIR=./results/${POOLING}/${LEVEL}/${SESSION}


python train_multitsk.py \
    --l2-reg 5e-6 \
    --lr 0.001 \
    --epochs 15 \
    --batch-size 64 \
    --filter-dims ${FILTERS} \
    --threshold ${THRESHOLD} \
    --pooling ${POOLING} \
    --level ${LEVEL} \
    --results_dir ./results/${OUTDIR}

