#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embed input C𝛼 distance matrices in structure space.
"""

import pickle
import argparse
from pathlib import Path

import torch
import numpy as np

from biotoolbox import fasta_reader
from biotoolbox.dbutils import MemoryMappedDatasetWriter

from gae.models import GAE
from gae.models import Embedding
from gae.loader import load_domain_list
from gae.loader import load_fasta, seq2onehot


# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Exists(p):
    p = Path(p)
    if not p.exists():
        raise IOError(f"{p} doesn\'t exist.")
    return p

def Nat(i):
    i = int(i)
    if i <= 0:
        raise ValueError(f"{i} isn\'t a natural number.")
    return i

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-list", required=True, type=Exists,
                        dest='inputs', help="Input file with list of absolute paths to test matrices.")
    parser.add_argument("-f", "--fasta", dest='fasta', required=True, type=Exists,
                        help="Maps ID to sequence")
    parser.add_argument('-M', '--model-file', type=Exists,
                        default='GAE_model', help="Name of the GAE model to be loaded.", required=True,
                        dest='model_name')
    parser.add_argument("-o", "--output-file", type=Path,
                        dest='outputs', help="Output location to dump embeddings.") 
    parser.add_argument("--memmap", action='store_true', default=False, 
                        help="Write down embeddings as a memory mapped database rather than separate npz files") 
    parser.add_argument('-d', '--filter-dims', dest='filters', type=Nat,
                        default=[64, 64, 64, 64, 64], nargs='+', help="Dimensions of GCN filters.")
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false')

    return parser.parse_args()

def load_model(model_file,
               filters=[64, 64, 64, 64, 64]):
    """Load pretrained GAE model"""
    gae = GAE(in_features=22, out_features=filters[-1], filters=filters, device=device)
    gae.load_state_dict(torch.load(model_file))
    gae.to(device)
    gae.eval()
    F = Embedding(gae)
    return F

def load_contact_map(tensor_file, resolution=10.):
    """
    Load a distance matrix saved as a torch tensor. Apply
    standard preprocessing to convert to contact map. 
    args:
        :tensor_file (Path or str)
        :resolution  (float) -- angstrom threshold for contact
    """
    tensor = torch.load(tensor_file).float()
    tensor[tensor <= resolution] = 1.
    tensor[tensor > resolution]  = 0.
    tensor = tensor - torch.diag(torch.diagonal(tensor))
    tensor = tensor + torch.eye(tensor.shape[1])
    tensor = tensor.view(-1, *tensor.numpy().shape)
    tensor = tensor.to(device)
    return tensor

def preprocess_sequence(seq):
    S = seq2onehot(seq)
    S = S.reshape(1, *S.shape)
    S = torch.from_numpy(S).float()
    S = S.to(device) 
    return S

def save_embedding(name, embedding, database):
    if isinstance(database, MemoryMappedDatasetWriter):
        database.set(name, embedding)
    elif isinstance(database, (str, Path)):
        database = Path(database)
        path = database / f"{name}.npz" 
        np.savez_compressed(path,embedding=embedding)
    else:
        raise ValueError(f"Can't record embeddings to a {type(database)}")

def seqdict(fastafile):
    return {k.lstrip('>'):v for k, v in fasta_reader(fastafile)}

if __name__ == '__main__':
    args = arguments()
    F      = load_model(args.model_name, filters=args.filters)
    id2seq = seqdict(args.fasta)

    with open(args.inputs, 'r') as f:
        lines = f.readlines()
        N = len(lines)
    paths = map(lambda line: Path(line.strip()), lines)
    if args.memmap:
        M = MemoryMappedDatasetWriter(args.outputs,
                                      embedding_dim=sum(args.filters),
                                      start=True,
                                      shard_size=N)
    else:
        M = args.outputs
        M.mkdir(exist_ok=True, parents=True)

    skip = N // 1000
    clear = f"\r{80 * ' '}\r"

    try:
        for i, tensor_file in enumerate(paths):
            structure_id = tensor_file.stem
            A = load_contact_map(tensor_file)
            S = preprocess_sequence(id2seq[structure_id])
            x = F((A, S))[0].cpu().detach().numpy()
            print(x)
            save_embedding(structure_id, x, M) 
            if args.verbose and ((i and not i % skip) or not i % N):
                print(f"{clear}{i}/{N}", end='', flush=True)

        print(f"{clear}Done! ({args.outputs})")
    except KeyboardInterrupt:
        print(f"{clear}Exiting due to user input")
    finally:
        if M is not None:
            M.close()
