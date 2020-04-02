#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use FAISS-clustering KMeans to cluster a database of protein embeddings.
"""
import argparse
from pathlib import Path
from datetime import datetime

import faiss
import joblib
from sklearn.neighbors import KDTree

from biotoolbox.dbutils import MemoryMappedDatasetReader 

def arguments():
    parser = argparse.ArgumentParser(description="Generate a KNN index.")
    parser.add_argument("input_database", help="Input data.", type=Path)
    parser.add_argument("--size", help="Dataset size (large := over 500k, small := under 500k",
                        choices=['large', 'small'], default='small')
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()

    reader = MemoryMappedDatasetReader(args.input_database, start=True)
    n, d = reader.shape
    start = datetime.now()
    print(f"Starting at {start}", flush=True)
    if args.size == 'large':
        outputfile = args.input_database / "trained.faiss.index"
        index  = faiss.index_factory(d, "OPQ64_128,IVF262144_HNSW32,PQ64")
        #index  = faiss.index_factory(d, "OPQ64_128,IVF16384_HNSW32,PQ64")
        #index  = faiss.index_factory(d, "OPQ64_128,IVF8192_HNSW32,PQ64")
        ivf = faiss.extract_index_ivf(index)
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(64))
        ivf.clustering_index = clustering_index
        print(f"Sent clustering index to GPU", flush=True)
        
        print(f"Clustering ({n}, {d}) matrix ({args.size} mode)", flush=True)
        index.train(reader.embedding_matrix)
        print("Finished training", flush=True)
        faiss.write_index(index, str(outputfile))
    else:
        outputfile = args.input_database / "trained.kdtree.index"
        print(f"Clustering ({n}, {d}) matrix ({args.size} mode)", flush=True)
        kdt = KDTree(reader.embedding_matrix, metric='euclidean') 
        print("Finished training")
        joblib.dump(kdt, str(outputfile))

    print(f"Wrote index to {outputfile}", flush=True)

    end = datetime.now()
    print(f"Finishing at {end} ({end - start} elapsed)", flush=True)
