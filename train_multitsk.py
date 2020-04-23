#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import sys
import pickle
import shutil
import argparse
from pathlib import Path

# numeric / third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    def __init__(self, domain_IDs, domain_class_map):
        'Initialization'
        self.domain_IDs = domain_IDs
        self.class_map  = domain_class_map

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
        A[A <= 10.0] = 1.0
        A[A > 10.0] = 0.0
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
        c_ = torch.zeros((len(batch),n_classes), dtype=torch.long)
        # padd
        for i in range(len(batch)):
            A_padded[i][:lengths[i], :][:, :lengths[i]] = batch[i][0]
            S_padded[i][:lengths[i], :] = batch[i][1]
            c_idx = batch[i][2]
            c_[i][c_idx] = 1.
        return (A_padded, S_padded, c_)
    return collate_padd

class MultitaskGAE(nn.Module):
    """
    Graph Autoencoder with a stack of Graph Convolution Layers and ZZ^T decoder.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 n_classes,
                 filters=[64, 64, 64],
                 bias=False,
                 adj_sq=False,
                 scale_identity=False,
                 device=None):
        super(MultitaskGAE, self).__init__()
        
        self.n_classes = n_classes
        self.filters   = filters

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
        self.cmap_decoder = InnerProductDecoder(in_features=sum(self.filters), out_features=out_features)
        self.classifier   = nn.Linear(sum(self.filters), out_features=self.n_classes)
        
        #self.seq_decoder = nn.Sequential(nn.Linear(sum(self.filters), out_features=in_features), nn.LogSoftmax(dim=-1))

    def forward(self, data):
        A, S = data
        gcn_embedd = [nn.Sequential(*list(self.cmap_encoder.children())[:i+1])((A, S))[1] for i in range(0, len(self.filters))]
        x = torch.cat(gcn_embedd, -1)
        cmap_out  = self.cmap_decoder(x)
        class_out = self.classifier(x)

        return cmap_out, class_out



def make_train_step(model, loss_bc, loss_nll, optimizer):
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
    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        cmap_hat, class_hat = model(x)
        loss = loss_bc(cmap_hat, y[0]) + loss_nll(class_hat, y[1])
        #loss = loss_bc(cmap_hat, y[0]) + loss_nll(seq_hat.view(-1, 22), torch.argmax(y[1], dim=2).view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def make_valid_step(model, loss_bc, loss_nll):
    """
    Builds function that performs a step in the valid loop
    args:
        :model - model to be trained
        :loss_bc - contact prediction loss
        :loss_nll - cath class prediction loss
    returns:
        :(callable) - train step function


    """
    def valid_step(x, y):
        model.eval()
        cmap_hat, class_hat = model(x)
        loss = loss_bc(cmap_hat, y[0]) + loss_nll(class_hat, y[1])
        #loss = loss_bc(cmap_hat, y[0]) + loss_nll(seq_hat.view(-1, 22), torch.argmax(y[1], dim=2).view(-1))

        return loss.item()

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

    parser.add_argument('--results_dir', type=Path, default='./results/', help="Directory to dump results and models.")
    parser.add_argument("--lists", type=Path, default=None, nargs=3, help="train, validation, and test paths in that order")

    parser.set_defaults(cuda=True)
    return parser.parse_args()

def generate_lists(domains):
    np.random.seed(104)
    np.random.shuffle(domains)

    p_tr, p_va = 0.75,  0.15
    train_idx = int(p_tr*len(domains))
    val_idx   = int(p_va*train_idx)

    print(train_idx - val_idx, train_idx, len(domains), flush=True) 

    train_list = domains[:train_idx - val_idx]
    valid_list = domains[train_idx - val_idx:train_idx]
    test_list  = domains[train_idx:]
    return train_list, valid_list, test_list


def plot_losses(title, filename, train_loss, valid_loss):
    train_avg, train_std = zip(*train_loss)
    valid_avg, valid_std = zip(*valid_loss)
    
    train_avg, train_std = map(np.array, (train_avg, train_std))
    valid_avg, valid_std = map(np.array, (valid_avg, valid_std))
    
    plt.figure()
    plt.fill_between(train_avg, train_avg - train_std, train_avg + train_std, alpha=0.25, color='C0')
    plt.plot(train_avg, '-', color='C0', label='Validation')

    plt.fill_between(valid_avg, valid_avg - valid_std, valid_avg + valid_std, alpha=0.25, color='C1')
    plt.plot(train_avg, '-', color='C1', label='Train')

    plt.title(title)
    plt.ylabel('(average) loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":
    path = '/mnt/ceph/users/dberenberg/Data/cath/'

    # load entire list
    domain2seqres = load_fasta(Path(path) / 'materials' / 'cath-dataset-nonredundant-S40.fa')
    domains = load_domain_list(Path(path) / 'materials' / 'annotated-cath-S40-domains.list')

    # make cath annotation map 
    cath_annotation_frame = pd.read_table(Path(path) / 'materials' / 'metadata' / 'domain-classifications.tsv')
    cath_topologies = sorted(cath_annotation_frame.TOPOL.unique())
    cath_class_map  = dict(zip(cath_topologies, range(cath_topologies.__len__())))

    cath_annotation_frame['class'] = cath_annotation_frame.TOPOL.apply(cath_class_map.get)
    cath_classifications = dict(cath_annotation_frame[['DOMAIN', 'class']].values)

    args = arguments()
    args.results_dir.mkdir(exist_ok=True, parents=True)
    args.models_dir = args.results_dir / 'models'
    args.models_dir.mkdir(exist_ok=True, parents=True)

    
    # CUDA for PyTorch
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # making/recording train/validation/test tests
    if not args.lists:
        args.train_list, args.valid_list, args.test_list = generate_lists(domains)
    else:
        args.train_list, args.valid_list, args.test_list = map(load_set, args.lists)

    for s in ['train', 'valid', 'test']:
        save_set(args.results_dir / f"{s}.list", getattr(args, f"{s}_list"))

    # build model
    gae = MultitaskGAE(in_features=22,
                       out_features=args.filter_dims[-1],
                       n_classes=len(cath_class_map),
                       filters=args.filter_dims, device=device)
    gae.to(device)

    # optimizer
    optimizer = optim.Adam(gae.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    # loss functions
    loss_bc  = nn.BCELoss() # classify contact
    loss_nll = nn.CrossEntropyLoss() # classify cath class
    loss_bc.cuda()
    loss_nll.cuda()

    # parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'collate_fn':make_collate_padd(len(cath_topologies))}

    # train generator
    training_dataset = Dataset(args.train_list, cath_classifications)
    training_generator = data.DataLoader(training_dataset, **params)

    # valid generator
    validation_dataset = Dataset(args.valid_list, cath_classifications)
    validation_generator = data.DataLoader(validation_dataset, **params)

    # Creates the train_step function
    train_step = make_train_step(gae, loss_bc, loss_nll, optimizer)

    # Creates the valid_step function
    valid_step = make_valid_step(gae, loss_bc, loss_nll)

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
    for epoch in range(1, args.epochs + 1):
        epoch_train_loss = []
        for batch_idx, (A_batch, S_batch, C_batch) in enumerate(training_generator):
            A_batch = A_batch.to(device)
            S_batch = S_batch.to(device)
            C_batch = C_batch.to(device)

            loss = train_step((A_batch, S_batch), (A_batch, C_batch))
            epoch_train_loss.append(loss)

            # print statistics
            if not (batch_idx % args.log_interval):
                progress = batch_idx * args.batch_size
                pct = 100. * batch_idx / T
                print(f"[*] (train) epoch {epoch} [{progress:5d}/{N:5d} ({pct:5.2f}%)]\tloss: {loss:0.6f}", flush=True)

        avg_train_loss = np.mean(epoch_train_loss)
        std_train_loss = np.std(epoch_train_loss)
        print(f"[!] E{epoch:3d}) average training loss: {avg_train_loss:0.4f}, std training loss: {std_train_loss:0.4f}", flush=True)
        sys.stdout.flush()
        train_loss_list.append((avg_train_loss, std_train_loss))
        with torch.no_grad():
            epoch_valid_loss = []
            for batch_idx, (A_batch, S_batch, C_batch) in enumerate(validation_generator):
                A_batch = A_batch.to(device)
                S_batch = S_batch.to(device)
                C_batch = C_batch.to(device)

                loss = valid_step((A_batch, S_batch), (A_batch, C_batch))
                epoch_valid_loss.append(loss)

        avg_valid_loss = np.mean(epoch_valid_loss)
        std_valid_loss = np.std(epoch_valid_loss)
        if avg_valid_loss < best_valid_loss:
            # save the model
            best_model_name = model_name_fmt.format(valid_loss=avg_valid_loss)
            torch.save(gae.state_dict(), args.models_dir + best_model_name + '_model.pt')

        print(f"[!] E{epoch:3d}) average validation loss: {avg_valid_loss:0.4f}, std validation loss: {std_valid_loss:0.4f}", flush=True)
        sys.stdout.flush()
        valid_loss_list.append((avg_valid_loss, std_valid_loss))

    # plot lossess
    title = f"Model loss: {filter_string}"
    plot_losses(title, args.results_dir + 'model_loss.png', train_loss_list, valid_loss_list)

    # save model
    shutil.copyfile(args.models_dir + best_model_name + '_model.pt', args.results_dir + 'final.pt')
    # record the losses
    with open(args.results_dir + 'loss.tsv', 'w') as lossfile:
        print("epoch", "train","train_std", "val", "val_std", sep='\t', file=lossfile)
        for epoch, ((tavg, tstd), (vavg, vstd)) in enumerate(zip(train_loss_list, valid_loss_list)):
            print(epoch, tavg, tstd, vavg, vstd, sep='\t', file=lossfile)

