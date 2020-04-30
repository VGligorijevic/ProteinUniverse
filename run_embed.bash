#!/usr/bin/env sh

#SBATCH -N1 
#SBATCH -p gpu 
#SBATCH --gres=gpu:01
#SBATCH --constraint=v100
#SBATCH --mem 128GB
#SBATCH --job-name emb

cd ${HOME}/projects/ProteinUniverse
source ${HOME}/anaconda3/etc/profile.d/conda.sh
source ${HOME}/.node_allocation/load_gpu_modules
conda activate protuniv

set -x
#LEVEL=C
#LEVEL=A
LEVEL=T

#ARCH="64-64-64-64-64"
#ARCH="64-128-128-256"
ARCH="128-128-128-128-128"
#ARCH="64-64-64"
#ARCH="64-128-128-256"

#THRESHOLD=10
#THRESHOLD=8
THRESHOLD=6
#THRESHOLD=4

SESSION=results/multitask/max/$LEVEL/${ARCH}__L${LEVEL}__T${THRESHOLD}

FASTA=Data/cath/materials/cath-dataset-nonredundant-S40.fa
STEM=$(basename "${SESSION}")
OUTPUT=Data/cath/databases/${STEM}
if [ -d $OUTPUT ]; then
    rm -r $OUTPUT
fi
LOC=/mnt/home/dberenberg/projects/ensemble-function-prediction/data/distance_maps/tensors/ca
LIST=/mnt/home/dberenberg/projects/ensemble-function-prediction/swiss_ids.list

OUTPUT=/mnt/home/dberenberg/projects/ensemble-function-prediction/data/efp-embeddings-mtgae_128x5.full_raw.npz

#FASTA=/mnt/home/dberenberg/projects/ensemble-function-prediction/data/human-swissmodel-STRING.fasta



python embed.py -i ${SESSION}/test.list -M ${SESSION} -o ${OUTPUT} -f ${FASTA} 
#python embed.py -i ${LIST} -M ${SESSION} -o ${OUTPUT} -f ${FASTA} -L ${LOC} 

