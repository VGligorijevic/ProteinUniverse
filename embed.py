#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embed input C𝛼 distance matrices in structure space.
"""

import pickle
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.manifold import TSNE

from biotoolbox import fasta_reader

from train_multitsk import MultitaskGAE, Embedding
#from gae.models import Embedding, GAE
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
    parser.add_argument('-M', '--model-session', type=Exists, help="Location of completed session", required=True,
                        dest='sess')
    parser.add_argument("-o", "--out", type=Path,
                        dest='output', help="Output .npz file containing 'E' matrix and 'id' array")
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false')

    parser.set_defaults(verbose=True)
    return parser.parse_args()

def load_model(model_file,
               n_classes=4,
               filters=[64, 64, 64, 64, 64]):
    """Load pretrained GAE model"""
    #gae = GAE(in_features=22, out_features=filters[-1], filters=filters, device=device)
    gae = MultitaskGAE(in_features=22, out_features=filters[-1], filters=filters, n_classes=n_classes, device=device)
    gae.load_state_dict(torch.load(model_file), strict=False)
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
    S = seq2onehot(seq, sub='X')
    S = S.reshape(1, *S.shape)
    S = torch.from_numpy(S).float()
    S = S.to(device)
    return S

def seqdict(fastafile):
    return {k.lstrip('>'):v for k, v in fasta_reader(fastafile)}

if __name__ == '__main__':
    path = '/mnt/ceph/users/dberenberg/Data/cath/'
    args = arguments()
    sess = args.sess
    args.model_name = sess / "final.pt"

    # retrieve model info
    with open(sess / 'params.txt', 'r') as params:
        level, threshold, arch_string, n_classes = filter(None, map(lambda line: line.strip(), params))  
    dimensions = list(map(int, arch_string.split()))
    threshold, n_classes = map(int, map(float, (threshold, n_classes)))

    F      = load_model(args.model_name, filters=dimensions, n_classes=n_classes)
    print(args.model_name, F.__class__.__name__)
    id2seq = load_fasta(args.fasta)

    with open(args.inputs, 'r') as f:
        lines = f.readlines()
        N = len(lines)

    paths = map(lambda stem: Path(path) / "cath-nr-S40_tensors" / f"{stem.strip()}.pt" , lines)
    
    M = args.output
    
    ##### silly display parameters ######
    mag  = 10000 ########################
    while not N // mag:################## 
        mag //= 10 ######################
    skip  = N // mag ####################
    clear = f"\r{80 * ' '}\r"############ 
    #####################################
  
    start = datetime.now()
    try:
        # sacrifice one for the grace of many
        fst_pth = next(paths)
        structure_id = fst_pth.stem
        A, S = load_contact_map(fst_pth, resolution=threshold), preprocess_sequence(id2seq[structure_id])
        x    = F((A,S))[0].cpu().detach().numpy() 
        x = np.max(x, axis=0)
        d = x.shape[0] # recover the width
        
        # create the big husker
        emat          = np.zeros((N, d))
        structure_ids = []
        emat[0] = x
        structure_ids.append(structure_id)
                                            # emat[0] is not forgotten
        for i, tensor_file in enumerate(paths, 1):
            structure_id = tensor_file.stem
            structure_ids.append(structure_id)

            A = load_contact_map(tensor_file, resolution=threshold)
            S = preprocess_sequence(id2seq[structure_id])
            # extract features
            x = F((A, S))[0].cpu().detach().numpy()
            x = np.max(x, axis=0)
            emat[i] = x
          # if we are printing msgs   and either on printerval or finished
            if all((args.verbose, i)) and any((not i % skip, not i % N)):
                print(f"{clear}[*] {i}/{N} ({datetime.now() - start} elapsed)", end='', flush=True)

        print(f"{clear}[!] Finished embedding.")
        print("[!] Running TSNE")
        R = TSNE(n_components=2).fit_transform(emat)
        np.savez_compressed(M, E=emat, id=np.array(structure_ids), R=R)

    except KeyboardInterrupt:
        print(f"{clear}Exiting due to user input")
    finally:
        elapsed = datetime.now() - start
        print(f"{clear}Done! ({args.output}) [{N}, {d}] ({elapsed} elapsed)")
