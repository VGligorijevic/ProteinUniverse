#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import sys
import csv
import pickle
import shutil
import argparse
import itertools
from pathlib import Path
from collections import Counter

# numeric / third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

# local
from gae.layers import GraphConv, MultiGraphConv, InnerProductDecoder
from gae.models import Embedding
from gae.loader import (load_domain_list, load_fasta,
                        seq2onehot, onehot2seq)

plt.switch_backend('agg')

    
class Dataset(data.Dataset):
    """ Characterizes a dataset for PyTorch """
    def __init__(self, domain_IDs, domain_class_map, threshold=10.):
        'Initialization'
        self.domain_IDs = domain_IDs
        self.class_map  = domain_class_map
        self.threshold  = threshold

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.domain_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.domain_IDs[index]

        # Load data
        A = torch.load(path + 'cath-nr-S40_tensors/' + ID + '.pt')
        S = torch.from_numpy(seq2onehot(domain2seqres[ID])).float()

        cls_idx = self.class_map[ID]

        # Create contact maps (10A cutoff)
        A[A <= self.threshold] = 1.0
        A[A >  self.threshold] = 0.0
        A = A - torch.diag(torch.diagonal(A))
        A = A + torch.eye(A.shape[1])

        return (A, S, cls_idx)

def make_collate_padd(n_classes):
    def collate_padd(batch):
        """
        Padds matrices of variable length
        """
        # get sequence lengths
        lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
        max_len = max(lengths)
        A_padded = torch.zeros((len(batch), max_len, max_len))
        S_padded = torch.zeros((len(batch), max_len, 22))
        S_padded[:, :, 21] = 1
        c_ = torch.zeros((len(batch),), dtype=torch.long)
        # padd
        for i in range(len(batch)):
            A_padded[i][:lengths[i], :][:, :lengths[i]] = batch[i][0]
            S_padded[i][:lengths[i], :] = batch[i][1]
            c_idx = batch[i][2]
            c_[i] = c_idx
        return (A_padded, S_padded, c_)
    return collate_padd

class MultitaskGAE(nn.Module):
    """
    Graph Autoencoder with a stack of Graph Convolution Layers and ZZ^T decoder.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 n_classes=None,
                 decode_seq=True,
                 pooling='sum',
                 filters=[64, 64, 64],
                 bias=False,
                 adj_sq=False,
                 scale_identity=False,
                 drop_prob=0.25,
                 device=None):
        super(MultitaskGAE, self).__init__()
        
        self.n_classes = n_classes
        self.decode_seq = decode_seq
        self.filters   = filters
        self.pooling = pooling

        # Sequence embedding
        self.seq_embedd = nn.Sequential(nn.Linear(in_features, self.filters[0]), nn.ReLU(inplace=True))

        # Encoder
        self.cmap_encoder = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else self.filters[layer - 1],
                                                       out_features=f,
                                                       bnorm=True,
                                                       activation=nn.ReLU(inplace=True),
                                                       bias=bias,
                                                       adj_sq=adj_sq,
                                                       scale_identity=scale_identity,
                                                       device=device) for layer, f in enumerate(self.filters)]))
        """
        self.cmap_encoder = nn.Sequential(*([MultiGraphConv(in_features=self.filters[0] if layer == 0 else self.filters[layer - 1],
                                                            out_features=f,
                                                            bnorm=True,
                                                            activation=nn.ReLU(inplace=True),
                                                            bias=bias,
                                                            device=device) for layer, f in enumerate(self.filters)]))
        """

        # Decoder
        k = 1 if self.pooling in ['max', 'sum'] else 2
        indim = k * sum(self.filters)
        self.cmap_decoder = InnerProductDecoder(in_features=sum(self.filters), out_features=out_features)
        
        if self.n_classes is not None:
            self.classification_branch = nn.Sequential(
                        nn.Linear(indim, out_features=self.n_classes),
                        nn.LogSoftmax(dim=-1)
                    )
        if self.decode_seq:
            self.seq_decoder = nn.Sequential(nn.Linear(sum(self.filters), out_features=in_features), nn.LogSoftmax(dim=-1))

    def forward(self, data):
        A, S = data
        gcn_embedd = [nn.Sequential(*list(self.cmap_encoder.children())[:i+1])((A, S))[1] for i in range(0, len(self.filters))]

        # residue-wise embeddings
        x = torch.cat(gcn_embedd, -1)

        cmap_out  = self.cmap_decoder(x)

        #reduc = x.sum(axis=1)
        if self.pooling == 'sum':
            comb = x.sum(dim=1)
        elif self.pooling == 'max':
            comb, inds = torch.max(x, 1) 
        elif self.pooling == 'concat':
            maxes, _ = torch.max(x, 1)
            sums     = x.sum(dim=1)
            comb = torch.cat([maxes, sums], 1)

        if self.n_classes is not None:
            class_out = self.classification_branch(comb)
        else:
            class_out = None
        if self.decode_seq:
            seq_out   = self.seq_decoder(x)
        else:
            seq_out = None

        return cmap_out, seq_out, class_out

class Embedding(nn.Module):
    """
    Extracting embeddings from the GraphConv layers of a pre-trained model.
    """
    def __init__(self, original_model):
        super(Embedding, self).__init__()
        self.pooling = original_model.pooling
        self.cmap_encoder = original_model.cmap_encoder
        self.num = len(list(self.cmap_encoder))
        self.gcn_layers = [self.cmap_encoder[:i] for i in range(1, self.num + 1)]

    def forward(self, x):
        A, S = x
        gcn_embedd = [nn.Sequential(*list(self.cmap_encoder.children())[:i+1])((A, S))[1] for i in range(self.num)]
        x = torch.cat(gcn_embedd, -1)
        if self.pooling == 'sum':
            comb = x.sum(axis=1)
        elif self.pooling == 'max':
            comb, inds = torch.max(x, 1)
        elif self.pooling == 'concat':
            maxes, _ = torch.max(x, 1)
            sums     = x.sum(dim=1)
            comb = torch.cat([maxes, sums], dim=0)
        return comb 



def make_train_step(model, loss_bc, loss_nll_seq, loss_nll_cath, optimizer):
    """
    Builds function that performs a step in the train loop
    args:
        :model - model to be trained
        :loss_bc - contact prediction loss
        :loss_nll - cath class prediction loss
        :optimizer - pytorch optimizer
    returns:
        :(callable) - train step function
    """
    lossnames = ['contact', 'seq', 'cath']
    losses = {'contact': loss_bc, 'seq':loss_nll_seq, 'cath':loss_nll_cath}

    def train_step(x, y):
        model.train()
        loss_step = dict()
        optimizer.zero_grad()
        yhat = model(x)
        
        returns = dict(zip(lossnames, (None, None, None)))
        
        for hat, actual, lname in zip(yhat, y, lossnames):
            if losses[lname] is not None:
                if lname == 'seq':
                    hat    = hat.view(-1, 22)
                    actual = torch.argmax(actual, dim=2).view(-1)
                returns[lname] = losses[lname](hat, actual) 
        loss = sum(returns[l] for l in returns if returns[l] is not None)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        return {l:(returns[l].item() if returns[l] is not None else 0.) for l in returns} 
    return train_step

def make_valid_step(model, loss_bc, loss_nll_seq, loss_nll_cath):
    """
    Builds function that performs a step in the valid loop
    args:
        :model - model to be trained
        :loss_bc - contact prediction loss
        :loss_nll - cath class prediction loss
    returns:
        :(callable) - train step function


    """
    lossnames = ['contact', 'seq', 'cath']
    losses = {'contact': loss_bc, 'seq':loss_nll_seq, 'cath':loss_nll_cath}

    def valid_step(x, y):
        model.eval()
        yhat = model(x)

        returns = dict(zip(lossnames, (None, None, None)))

        for hat, actual, lname in zip(yhat, y, lossnames):
            if losses[lname] is not None:
                if lname == 'seq':
                    hat    = hat.view(-1, 22)
                    actual = torch.argmax(actual, dim=2).view(-1)
                returns[lname] = losses[lname](hat, actual) 
        loss = sum(returns[l] for l in returns if returns[l] is not None)
        return {l:(returns[l].item() if returns[l] is not None else 0.) for l in returns} 

    # Returns the function that will be called inside the valid loop
    return valid_step

def save_set(filename, S):
    with open(filename, 'w') as f:
        print(*S, sep='\n', file=f)

def load_set(filename):
    with open(filename, 'r') as f:
        return list(map(lambda line: line.strip(), f))

def NumericGreaterThan(bound, numeric_type):
    """Verifies a float > some number"""
    def verifier(x):
        x = numeric_type(x)
        if x <= bound:
            raise ValueError(f"{x} is not > {bound}")
        return x
    return verifier

Nat          = NumericGreaterThan(0, int)
NaturalFloat = NumericGreaterThan(0, float)

def arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='PyTorch GAE Training.')
    parser.add_argument('--cuda',
                        action='store_true',
                        help="Enables CUDA training.",
                        dest='cuda')

    parser.add_argument("--no-cuda",
                        action='store_false',
                        dest='cuda',
                        help="Disables CUDA training.")

    parser.add_argument('--log-interval',
                        type=Nat,
                        default=10,
                        help="How many batches to wait before logging training status.")

    parser.add_argument('-dims',
                        '--filter-dims',
                        type=Nat,
                        default=[64, 128],
                        nargs='+', 
                        help="Dimensions of GCN filters.")

    parser.add_argument('-l2','--l2-reg',
                        type=NaturalFloat,
                        default=1e-4,
                        help="L2 regularization coefficient.")

    parser.add_argument('--lr',
                        type=NaturalFloat,
                        default=0.001,
                        help="Initial learning rate.")

    parser.add_argument('--epochs',
                        type=Nat,
                        default=20,
                        help="Number of epochs to train.")

    parser.add_argument('--batch-size',
                        type=Nat,
                        default=64,
                        help="Batch size.")

    parser.add_argument("-l", "--level",
                        choices=list("CATH0"), default="C", help="CATH annotation level")

    parser.add_argument("-t", "--threshold",
                        dest='threshold', type=NaturalFloat, help="Aangstrom threshold", default=10.)

    parser.add_argument("-p", "--pooling",
                        dest="pool", type=str, choices=['max', 'sum', 'concat'], default='sum')

    parser.add_argument('--results_dir', type=Path, default='./results/', help="Directory to dump results and models.")
    parser.add_argument("--lists", type=Path, default=None, nargs=3, help="train, validation, and test paths in that order")

    parser.set_defaults(cuda=True)
    return parser.parse_args()

def generate_sets(annotations, stratifier=None, test_size=0.25, val_size=0.15, random_state=104):
    fst_stratifier = annotations[stratifier] if stratifier is not None else None
    trainval, test = train_test_split(annotations,
                                      test_size=test_size,
                                      stratify=fst_stratifier,
                                      random_state=random_state)

    snd_stratifier = trainval[stratifier] if stratifier is not None else None
    train, val     = train_test_split(trainval,
                                      test_size=val_size,
                                      stratify=snd_stratifier,
                                      random_state=random_state)

    return train.DOMAIN.values, val.DOMAIN.values, test.DOMAIN.values

def plot_losses(title, filename, train_loss, valid_loss):
    train_loss = [t['total'] for t in train_loss]
    valid_loss = [v['total'] for v in valid_loss]

    plt.figure()
    plt.plot(train_loss, '-', color='C0', label='Train')
    plt.plot(valid_loss, '-', color='C1', label='Validation')

    plt.title(title)
    plt.ylabel('(average) loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(filename, bbox_inches='tight')


def build_annotations(path, domains, level, cutoff=None, verbose=True):
    """
    Builds the annotations for a set of domains.
    args:
        :path (Path or str) - location of annotation file
        :domains (list)     - list of valid domains
        :level (str)        - column to use as label set
        :cutoff (int)       - minimum size of class
    returns:
        :(pd.DataFrame) - annotations
        :(dict)         - class map
    """
    cath_annotation_frame = pd.read_table(path)
    cath_annotation_frame = cath_annotation_frame[cath_annotation_frame.DOMAIN.isin(domains)].copy()

    vc = cath_annotation_frame[level].value_counts()
    if cutoff is None: cutoff = 0
    large = vc[vc >  cutoff].index

    is_large = cath_annotation_frame[level].isin(large)
    is_small = ~is_large

    # adjust classes
    col = f'adjusted_{level}'
    UNK = "zzzzz_UNKNOWN"
    cath_annotation_frame.loc[is_small, col] = UNK
    cath_annotation_frame.loc[is_large, col] = cath_annotation_frame.loc[is_large, level]

    # sort classes alphabetically
    cath_classes   = sorted(cath_annotation_frame[col].unique())
    cath_class_map = {c:i for i, c in enumerate(cath_classes)}

    cath_annotation_frame['class'] = cath_annotation_frame[col].apply(cath_class_map.get)
    cath_classifications = dict(cath_annotation_frame[['DOMAIN', 'class']].values)
    
    # build class weighting
    class_counts = Counter(cath_annotation_frame[col].values)
    weights = torch.zeros(len(class_counts))
    mx = max(class_counts.values())
    for cls in cath_classes:
        idx = cath_class_map[cls]
        ct  = class_counts[cls]
        weights[idx] = float(mx / ct)
    
    if UNK in cath_annotation_frame[col].unique():
        weights[-1] = 0.  # give no shit about unknowns 

    if verbose:
        print(f"# {level} classes = {len(cath_classes)}")
        print(dict(zip(cath_classes, weights)))

    return cath_annotation_frame, cath_classifications, weights 


if __name__ == "__main__":
    path = '/mnt/ceph/users/dberenberg/Data/cath/'

    args = arguments()
    args.results_dir.mkdir(exist_ok=True, parents=True)
    args.models_dir = args.results_dir / 'models'
    args.models_dir.mkdir(exist_ok=True, parents=True)


    # load entire list
    domain2seqres = load_fasta(Path(path) / 'materials' / 'cath-dataset-nonredundant-S40.fa')
    domains = load_domain_list(Path(path) / 'materials' / 'annotated-cath-S40-domains.list')
    
    ## CATH classification processing #######################
    table_loc = Path(path) / 'materials/metadata/domain-classifications.tsv'
    
    if args.level in "CATH":
        level = {"C":"CLASS", "A":"ARCH", "T":"TOPOL", "H":"HOMOL"}[args.level]
        cath_annotation_frame, cath_classifications, weights = build_annotations(table_loc, domains, level, cutoff=25)
    else:
        level = "CLASS"
        # fool em
        cath_annotation_frame, cath_classifications, weights = build_annotations(table_loc, domains, "CLASS", cutoff=25)

    with open(args.results_dir / 'params.txt', 'w') as par:
        astring = " ".join(map(str, args.filter_dims))
        print(level, args.threshold, astring,
              f"{len(weights) if args.level in 'CATH' else None}", args.pool, file=par, sep='\n')

        print(f"annotation_level={level}",
              f"contact_thresh={args.threshold}",
              f"architecture={astring}",
              f"pooling={args.pool}",
              f"n_classes={len(weights) if args.level in 'CATH' else None}", sep='\n')

    # CUDA for PyTorch
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # making/recording train/validation/test tests
    if not args.lists:
        args.train_list, args.valid_list, args.test_list = generate_sets(cath_annotation_frame,
                                                                         stratifier='class')
    else:
        args.train_list, args.valid_list, args.test_list = map(load_set, args.lists)
    for s in ['train', 'valid', 'test']:
        save_set(args.results_dir / f"{s}.list", getattr(args, f"{s}_list"))

    # build model
    gae = MultitaskGAE(in_features=22,
                       out_features=args.filter_dims[-1],
                       n_classes=len(weights) if args.level in "CATH" else None,
                       pooling=args.pool,
                       decode_seq=True,
                       filters=args.filter_dims, device=device)
    gae.to(device)

    # optimizer
    optimizer = optim.Adam(gae.parameters(), lr=args.lr, weight_decay=args.l2_reg)


    # loss functions
    loss_bc  = nn.BCELoss()                    # classify contact
    loss_bc.cuda()
    loss_nll_seq  = nn.NLLLoss()               # classify sequence
    loss_nll_seq.cuda()
    if args.level in "CATH":
        loss_nll_cath = nn.NLLLoss(weight=weights) # classify cath class
        loss_nll_cath.cuda()
    else:
        loss_nll_cath = None

    # parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'collate_fn':make_collate_padd(len(weights))}

    # train generator
    training_dataset = Dataset(args.train_list, cath_classifications, threshold=args.threshold)
    training_generator = data.DataLoader(training_dataset, **params)

    # valid generator
    validation_dataset = Dataset(args.valid_list, cath_classifications, threshold=args.threshold)
    validation_generator = data.DataLoader(validation_dataset, **params)

    # Creates the train_step function
    train_step = make_train_step(gae, loss_bc, loss_nll_seq, loss_nll_cath, optimizer)

    # Creates the valid_step function
    valid_step = make_valid_step(gae, loss_bc, loss_nll_seq, loss_nll_cath)

    # mini-batch update
    train_loss_list = []
    valid_loss_list = []
    
    # save current best models.
    best_valid_loss = np.inf
    best_model_name = None
    filter_string = "-".join(map(str, args.filter_dims))
    model_name_fmt  = f"{gae.__class__.__name__}__f{filter_string}""__vloss{valid_loss:0.4f}.pt"
    

    args.results_dir = str(args.results_dir) + "/"
    args.models_dir  = str(args.models_dir) + "/"

    T, V = map(len, (training_generator, validation_generator)) 
    N = len(training_generator.dataset)
    print("starting")
    for epoch in range(1, args.epochs + 1):
        losses = dict(total=0., contact=0., seq=0., cath=0.)

        for batch_idx, (A_batch, S_batch, C_batch) in enumerate(training_generator):
            A_batch = A_batch.to(device)
            S_batch = S_batch.to(device)
            C_batch = C_batch.to(device)

            loss_dictionary = train_step((A_batch, S_batch), (A_batch, S_batch, C_batch))
            for loss in ['contact', 'seq', 'cath']:
                losses['total'] += loss_dictionary[loss]
                losses[loss] += loss_dictionary[loss]

            # print statistics
            if not (batch_idx % args.log_interval):
                progress = batch_idx * args.batch_size
                pct = 100. * batch_idx / T
                loss_summary = ", ".join([f"{k}={v:0.5f}" for k, v in loss_dictionary.items()])
                fmt = f"[*] (train) E{epoch} [{progress:5d}/{N:5d} ({pct:5.2f}%)]\t" + loss_summary
                print(fmt, flush=True)

        print(f"[!] E{epoch:3d}) (train) Averages:", flush=True)
        for loss in losses:
            print(f"\t{loss}: {losses[loss] / len(training_generator):0.4f}", flush=True)

        train_loss_list.append({loss: losses[loss] / len(training_generator) for loss in losses})

        with torch.no_grad():
            valid_losses = dict(total=0., contact=0., seq=0., cath=0.)
            for batch_idx, (A_batch, S_batch, C_batch) in enumerate(validation_generator):
                A_batch = A_batch.to(device)
                S_batch = S_batch.to(device)
                C_batch = C_batch.to(device)

                loss_dictionary = valid_step((A_batch, S_batch), (A_batch, S_batch, C_batch))
                for loss in ['contact', 'seq', 'cath']:
                    valid_losses['total'] += loss_dictionary[loss]
                    valid_losses[loss] += loss_dictionary[loss]

        avg_valid_loss = valid_losses['total'] / len(validation_generator)
        if avg_valid_loss < best_valid_loss:
            # save the model
            best_model_name = model_name_fmt.format(valid_loss=avg_valid_loss)
            torch.save(gae.state_dict(), args.models_dir + best_model_name + '_model.pt')

        print(f"[!] E{epoch:3d}) (validation) Averages:", flush=True)
        for loss in losses:
            print(f"\t{loss}: {valid_losses[loss] / len(validation_generator):0.4f}", flush=True)
        valid_loss_list.append({loss: valid_losses[loss] / len(validation_generator) for loss in valid_losses})

    # plot lossess
    title = f"Model loss: {filter_string} (threshold = {args.threshold}"
    plot_losses(title, args.results_dir + 'model_loss.png', train_loss_list, valid_loss_list)

    # save model
    shutil.copyfile(args.models_dir + best_model_name + '_model.pt', args.results_dir + 'final.pt')
    # record the losses
    with open(args.results_dir + 'loss.tsv', 'w') as lossfile:
        sets = ['train', 'valid']
        fields = ['seq', 'total', 'cath', 'contact']
        columns = [f"{s}_{f}" for (s, f) in itertools.product(sets, fields)]

        writer = csv.DictWriter(lossfile, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        for tavg, vavg in zip(train_loss_list, valid_loss_list):
            row = {**{f'train_{k}': v for k,v in tavg.items()}, **{f'valid_{k}': v for k,v in vavg.items()}}
            writer.writerow(row)
