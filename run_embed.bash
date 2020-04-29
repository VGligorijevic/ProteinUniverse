#!/usr/bin/env sh

#SBATCH -N1 
#SBATCH -p gpu 
#SBATCH --gres=gpu:01
#SBATCH --constraint=v100
#SBATCH --job-name emb

cd ${HOME}/projects/ProteinUniverse
source ${HOME}/anaconda3/etc/profile.d/conda.sh
source ${HOME}/.node_allocation/load_gpu_modules
conda activate protuniv

set -x
#LEVEL=C
LEVEL=A
#ARCH="128-128-128-128-128"
#ARCH="64-64-64-64-64"
#ARCH="64-64-64"
ARCH="64-128-128-256"

#THRESHOLD=10
#THRESHOLD=8
#THRESHOLD=6
THRESHOLD=4

SESSION=results/multitask/max/$LEVEL/${ARCH}__L${LEVEL}__T${THRESHOLD}

FASTA=Data/cath/materials/cath-dataset-nonredundant-S40.fa
STEM=$(basename "${SESSION}")
OUTPUT=Data/cath/databases/${STEM}
if [ -d $OUTPUT ]; then
    rm -r $OUTPUT
fi
python embed.py -i ${SESSION}/test.list -M ${SESSION} -o ${OUTPUT} -f ${FASTA}

