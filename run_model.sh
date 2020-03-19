# !/bin/bash
# Run GAE model

python train.py \
    --filter-dims 64 64 64 64 64 \
    --l2-reg 5e-6 \
    --lr 0.001 \
    --epochs 10 \
    --batch-size 64 \
    --results_dir ./results/ \
    --model-name GAE_64-64-64-64-64 \
