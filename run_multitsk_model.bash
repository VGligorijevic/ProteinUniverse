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

python train_multitsk.py \
    --filter-dims 64 64 64 64 64 \
    --l2-reg 5e-6 \
    --lr 0.001 \
    --epochs 10 \
    --batch-size 64 \
    --results_dir ./results/multitask/gae_64x5 #--model-name GAE_64-64-64-64-64 \
    #--lists lists/train.list lists/valid.list lists/test.list \
