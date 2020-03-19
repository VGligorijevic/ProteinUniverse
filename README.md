# ProteinUniverse

## Training
Use `run_model.sh` to train new models

## Data (CATH domains)
All the contact maps used for training of GAE are extracted from there:
`ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/`

## To extract AE embeddings use:

```
python compute_embeddings.py --model-name ./results/GAE_64-64-64-64-64_model.pt
--filter-dims 64 64 64 64 64 --out-pckl ./results/gae_cath_embeddings.pckl
```


## tSNE viz

```
python viz_embeddings.py ./results/gae_cath_embeddings.pckl
```

Produces: `*_vectors.tsv` and `*_metadata.tsv` files hat can be used by
`Embedding Projector (https://projector.tensorflow.org/)`
