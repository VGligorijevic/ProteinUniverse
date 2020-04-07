#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze clustering performance by
    - Sample 5000 members of the test set
    - Find their 500 nearest neighbors
    - For each sample:
        topol = Get_CATH_TOPOL sample 
        For each neighbor:
            score = TMAlign(sample, neighbor )
    - For k in [1 ... 500]:
        get proportion of in-fold neighbors vs out-fold neighbors
        
"""
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

from biotoolbox.dbutils import load_knn_db


PU = Path(__file__).parent
home = PU /  "integrity_analysis"
home.mkdir(exist_ok=True, parents=True)

test_set    = PU / "gae_64x5" / "test.list"
structures  = PU / "Data" / "cath" / "cath-dataset-nr-S40" 

topk_distances = home / "topk_distances.tsv"
tmalign_files  = home / "tmscores"
tmalign_files.mkdir(exist_ok=True, parents=True)

tmalign = (PU / "dependencies" / "bin" / "TMAlign").absolute()
tmalign_tasks = home / "compute_tmscores.tsk"

if __name__ == '__main__':
    N = 300
    k = 500

    sample_file = home / "sampled.list"
    if not sample_file.exists(): 
        with open(test_set, 'r') as infile, open(sample_file, 'w') as outfile:
            all_samples = list(map(lambda line: line.strip(), infile.readlines())) 
            samps = np.random.choice(all_samples, size=N)
            print(*samps, sep='\n', file=outfile)
    else:
        with open(sample_file, 'r') as f:
            samps = list(map(lambda line: line.strip(), f.readlines()))
    print(N, k, len(samps))

    db = load_knn_db(PU / "Data" / "cath" / "database")
    start = datetime.now()
    if not topk_distances.exists():
        with open(topk_distances, 'w') as tsv, open(tmalign_tasks:
            writer = csv.DictWriter(tsv, fieldnames=['query', 'neighbor', 'neighbor_rank', 'distance'], delimiter='\t')
            writer.writeheader()
            for i, samp in enumerate(samps):
                xq   = db.embedding(samp).reshape(1,-1)
                topk, distances = db.nearest_neighbors(xq, k=k)
                for rank, (neighb, dist) in enumerate(zip(topk, distances)):
                    row = dict(query=samp, neighbor=neighb, neighbor_rank=rank, distance=dist)
                    writer.writerow(row)
                print(f"{samp} ({i}/{N}) ({datetime.now() - start} elapsed)")
        print(f"Done! ({datetime.now() - start} elapsed)")
    print("done")
