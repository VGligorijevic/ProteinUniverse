#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtin
import sys
import pickle
import argparse
from pathlib import Path

# numeric / third-party
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

# local
from gae.models import GAE
from gae.models import Embedding
from gae.loader import load_domain_list, Dataset, collate_padd
from gae.loader import load_fasta, seq2onehot
from gae.loader import onehot2seq

plt.switch_backend('agg')

def plot_losses(train_loss, valid_loss):
    plt.figure()
    plt.plot(train_loss, '-')
    plt.plot(valid_loss, '-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(args.results_dir + args.model_name + '_model_loss.png', bbox_inches='tight')


def make_train_step(model, loss_bc, loss_nll, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        cmap_hat, seq_hat = model(x)
        loss = loss_bc(cmap_hat, y[0]) + loss_nll(seq_hat.view(-1, 22), torch.argmax(y[1], dim=2).view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def make_valid_step(model, loss_bc, loss_nll):
    # Builds function that performs a step in the valid loop
    def valid_step(x, y):
        model.eval()
        cmap_hat, seq_hat = model(x)
        loss = loss_bc(cmap_hat, y[0]) + loss_nll(seq_hat.view(-1, 22), torch.argmax(y[1], dim=2).view(-1))

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

    #parser.add_argument('--model-name', type=Path, default='GAE_model', help="Name of the GAE model to be saved.")
    #parser.add_argument('--train-list', type=str, default='train_domains.txt', help="List with train structure IDs.")
    #parser.add_argument('--test-list', type=str, default='test_domains.txt', help="List with test structure IDs.")

    parser.set_defaults(cuda=True)
    return parser.parse_args()



if __name__ == "__main__":
    path = '/mnt/ceph/users/dberenberg/Data/cath/'

    # load entire list
    domain2seqres = load_fasta(Path(path) / 'materials' / 'cath-dataset-nonredundant-S40.fa')
    domains = load_domain_list(Path(path) / 'materials' / 'annotated-cath-S40-domains.list')
    
    args = arguments()
    args.results_dir.mkdir(exist_ok=True, parents=True)
    
    # CUDA for PyTorch
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    if not args.lists:
        np.random.seed(104)
        np.random.shuffle(domains)

        p_tr = 0.75
        p_va = 0.15
        train_idx = int(p_tr*len(domains))
        val_idx   = int(p_va*train_idx)

        print(train_idx - val_idx, train_idx, len(domains), flush=True) 

        args.train_list = domains[:train_idx - val_idx]
        args.valid_list = domains[train_idx - val_idx:train_idx]
        args.test_list  = domains[train_idx:]
    else:
        trn, val, tes = map(load_set, args.lists)
        args.train_list = trn
        args.valid_list = val
        args.test_list  = tes

    save_set(args.results_dir / "train.list", args.train_list)
    save_set(args.results_dir / "valid.list", args.valid_list)
    save_set(args.results_dir / "test.list" , args.test_list)

    args.model_name = f"GAE__"+"-".join(map(str, args.filter_dims)) + f"__lr{args.lr}"
    args.model_name = args.model_name + f"__batch_size{args.batch_size}"
    args.model_name = args.model_name + f"__l2{args.l2_reg}"

    args.results_dir = str(args.results_dir) + "/"

    print(args.model_name, flush=True)
    gae = GAE(in_features=22, out_features=args.filter_dims[-1], filters=args.filter_dims, device=device)
    gae.to(device)

    # optimizer
    optimizer = optim.Adam(gae.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    # loss functions
    loss_bc = nn.BCELoss()
    loss_nll = nn.NLLLoss()
    loss_bc.cuda()
    loss_nll.cuda()

    # parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'collate_fn': collate_padd}

    # train generator
    training_dataset = Dataset(args.train_list)
    training_generator = data.DataLoader(training_dataset, **params)

    # valid generator
    validation_dataset = Dataset(args.valid_list)
    validation_generator = data.DataLoader(validation_dataset, **params)

    # Creates the train_step function
    train_step = make_train_step(gae, loss_bc, loss_nll, optimizer)

    # Creates the valid_step function
    valid_step = make_valid_step(gae, loss_bc, loss_nll)

    # mini-batch update
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        for batch_idx, (A_batch, S_batch) in enumerate(training_generator):
            A_batch = A_batch.to(device)
            S_batch = S_batch.to(device)

            loss = train_step((A_batch, S_batch), (A_batch, S_batch))
            train_loss += loss

            # print statistics
            if not (batch_idx % args.log_interval):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               batch_idx * args.batch_size, 
                                                                               len(training_generator.dataset),
                                                                               100. * batch_idx / len(training_generator),
                                                                               loss), flush=True)
        print('====> Epoch: {} Average train_loss: {:.4f}'.format(epoch, train_loss / len(training_generator)), flush=True)
        sys.stdout.flush()
        train_loss_list.append(train_loss/len(training_generator))
        with torch.no_grad():
            valid_loss = 0.0
            for batch_idx, (A_batch, S_batch) in enumerate(validation_generator):
                A_batch = A_batch.to(device)
                S_batch = S_batch.to(device)

                loss = valid_step((A_batch, S_batch), (A_batch, S_batch))
                valid_loss += loss
        print('====> Epoch: {} Average valid_loss: {:.4f}'.format(epoch, valid_loss / len(validation_generator)), flush=True)
        sys.stdout.flush()
        valid_loss_list.append(valid_loss/len(validation_generator))

    # plot lossess
    plot_losses(train_loss_list, valid_loss_list)

    # save model
    torch.save(gae.state_dict(), args.results_dir + args.model_name + '_model.pt')
