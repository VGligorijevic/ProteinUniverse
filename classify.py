#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd

from gae.loader import load_domain_list
from gae.loader import load_fasta, seq2onehot

from train_multitsk import MultitaskGAE, Embedding
from embed import load_contact_map, preprocess_sequence, Exists, Nat 

# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-list", required=True, type=Exists,
                        dest='inputs', help="Input file with list of absolute paths to test matrices.")
    parser.add_argument('-M', '--model-file', type=Exists,
                        default='GAE_model', help="Name of the GAE model to be loaded.", required=True,
                        dest='model_name')
    parser.add_argument("-o", "--output", type=Path,
                        dest='outputs', help="Output location to dump classifications.")
    parser.add_argument('-d', '--filter-dims', dest='filters', type=Nat,
                        default=[64, 64, 64, 64, 64], nargs='+', help="Dimensions of GCN filters.")
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false')

    parser.set_defaults(verbose=True)
    return parser.parse_args()

def load_model(model_file,
               filters=[64, 64, 64, 64, 64]):
    """Load pretrained GAE model"""
    #gae = GAE(in_features=22, out_features=filters[-1], filters=filters, device=device)
    gae = MultitaskGAE(in_features=22, out_features=filters[-1], filters=filters, n_classes=1377, device=device)
    gae.load_state_dict(torch.load(model_file), strict=False)
    gae.to(device)
    gae.eval()
    return gae

if __name__ == '__main__':
    path = '/mnt/ceph/users/dberenberg/Data/cath/'

    args = arguments()

    # make cath annotation map 
    cath_annotation_frame = pd.read_table(Path(path) / 'materials' / 'metadata' / 'domain-classifications.tsv')
    cath_topologies = sorted(cath_annotation_frame.TOPOL.unique())
    cath_class_map  = dict(zip(cath_topologies, range(cath_topologies.__len__())))

    cath_annotation_frame['class'] = cath_annotation_frame.TOPOL.apply(cath_class_map.get)
    cath_classifications = dict(cath_annotation_frame[['DOMAIN', 'class']].values)
    
    # load up model
    F    = load_model(args.model_name, filters=args.filters)

    # get sequence maps
    id2seq = load_fasta(Path(path) / 'materials' / 'cath-dataset-nonredundant-S40.fa')

    # get test domains
    with open(args.inputs, 'r') as f:
        lines = f.readlines()
        N = len(lines)
    paths = map(lambda line: Path(line.strip()), lines)

    mag  = 10000
    while not N // mag:
        mag //= 10

    skip = N // mag
    print(skip, mag, N, flush=True)
    clear = f"\r{80 * ' '}\r"

    print(F)
    start = datetime.now()
    try:
        with open(args.outputs, 'w') as outfile:
            columns = ['domain', 'prediction', 'actual']
            print(*columns, sep='\t', file=outfile)
            for i, tensor_file in enumerate(paths):
                structure_id = tensor_file.stem
                cath_class = cath_classifications.get(structure_id)

                A = load_contact_map(tensor_file)
                S = preprocess_sequence(id2seq[structure_id])
                class_hat = F((A, S))[1][0]
                pred_class = torch.argmax(class_hat).cpu().detach().numpy()
                print(structure_id, pred_class, cath_class, sep='\t', file=outfile)

                if all((args.verbose, i)) and any((not i % skip, not i % N)):
                    print(f"{clear}{i}/{N} ({datetime.now() - start} elapsed)", end='', flush=True)


    except KeyboardInterrupt:
        print(f"{clear}Exiting due to user input")
    finally:
        elapsed = datetime.now() - start
        print(f"{clear}Done! ({args.outputs}) ({elapsed} elapsed)")
