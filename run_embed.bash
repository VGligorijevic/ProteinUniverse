#!/usr/bin/env sh

#SBATCH -N1 
#SBATCH -p gpu 
#SBATCH --gres=gpu:01
#SBATCH --constraint=v100

cd ${HOME}/projects/ProteinUniverse
source ${HOME}/anaconda3/etc/profile.d/conda.sh
source ${HOME}/.node_allocation/load_gpu_modules
conda activate protuniv

set -x
#MODEL=results/gae_64x5/final.pt
#FILTERS="64 64 64 64 64"
# =====================================
#MODEL=results/gae_64x7/final.pt
#FILTERS="64 64 64 64 64 64 64"
# =====================================
#MODEL=results/gae_128x2__96x1__64x2/final.pt
#FILTERS="128 128 96 64 64"
# =====================================
#MODEL=results/gae_96x4/final.pt
#FILTERS="96 96 96 96"
# =====================================
MODEL=results/gae_128x4/final.pt
FILTERS="128 128 128 128"

FASTA=Data/cath/materials/cath-dataset-nonredundant-S40.fa
DIR=$(dirname "${MODEL}")
STEM=$(basename "${DIR}")
OUTPUT=Data/cath/databases/${STEM}
if [ -d $OUTPUT ]; then
    rm -r $OUTPUT
fi
mkdir -p $OUTPUT

python embed.py -i test-tensors.list -M ${MODEL} -o ${OUTPUT} -f ${FASTA} -d $FILTERS --memmap

