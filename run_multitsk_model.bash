#!/bin/bash
# Run GAE model

#SBATCH --constraint=v100
#SBATCH --partition gpu
#SBATCH --gres=gpu:01

cd $HOME/projects/ProteinUniverse
source $HOME/anaconda3/etc/profile.d/conda.sh
source $HOME/.node_allocation/load_gpu_modules

conda activate protuniv

python train_multitsk.py \
    --filter-dims 64 128 128 256 \
    --l2-reg 5e-6 \
    --lr 0.001 \
    --epochs 30 \
    --batch-size 64 \
    --lists lists/train.list lists/valid.list lists/test.list \
    --results_dir ./results/64-128-128-256 #--model-name GAE_64-64-64-64-64 \
