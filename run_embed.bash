#!/usr/bin/env sh

#SBATCH -N1 
#SBATCH -p gpu 
#SBATCH --gres=gpu:v100-32gb:01
#SBATCH --mem 128GB
#SBATCH --job-name emb

cd ${HOME}/projects/ProteinUniverse
source ${HOME}/anaconda3/etc/profile.d/conda.sh
source ${HOME}/.node_allocation/load_gpu_modules
conda activate protuniv

set -x
#LEVEL=C
#LEVEL=A
#LEVEL=T
#LEVEL=H
LEVEL=0

#ARCH="64-128-128-256"
ARCH="128-128-128-128"

THRESHOLD=6

#POOL="max"
#POOL="sum"
POOL="concat"

STEM=${ARCH}__L${LEVEL}__T${THRESHOLD}__P${POOL}

SESSION=./results/results/${POOL}/${LEVEL}/${STEM}


# Settings for CATH clustering
FASTA=Data/cath/materials/cath-dataset-nonredundant-S40.fa
OUTPUT=Data/cath/databases/SSC/${STEM}
LIST=${SESSION}/test.list
LOC=/mnt/home/dberenberg/projects/ProteinUniverse/Data/cath/cath-nr-S40_tensors/

# Settings for multimodal human fp
#OUTPUT=/mnt/home/dberenberg/projects/multimodal-fp/data/efp-embeddings-mtgae_${STEM}.full_raw.npz#FASTA=/mnt/home/dberenberg/projects/multimodal-fp/data/human-swissmodel-STRING.fasta
#FASTA=/mnt/home/dberenberg/projects/multimodal-fp/data/human-swissmodel-STRING.fasta
#LOC=/mnt/home/dberenberg/projects/multimodal-fp/data/distance_maps/tensors/ca
#LIST=/mnt/home/dberenberg/projects/multimodal-fp/swiss_ids.list

if [ -d $OUTPUT ]; then
    rm -r $OUTPUT
fi
mkdir -p ${OUTPUT}

python embed.py -i ${LIST} -M ${SESSION} -o ${OUTPUT} -f ${FASTA} -L ${LOC} 
