import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from gae.models import GAE
from gae.models import Embedding
from gae.loader import load_domain_list, Dataset, collate_padd
from gae.loader import load_fasta, seq2onehot
from gae.loader import onehot2seq

import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


path = '/mnt/ceph/users/vgligorijevic/ProteinDesign/CATH/'
domain2seqres = load_fasta(path + 'cath-dataset-nonredundant-S40.fa')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='PyTorch GAE Training.')
parser.add_argument('--no-cuda', action='store_true', default=False, help="Enables CUDA training.")
parser.add_argument('--log-interval', type=int, default=10, help="How many batches to wait before logging training status.")
parser.add_argument('-dims', '--filter-dims', type=int, default=[64, 128], nargs='+', help="Dimensions of GCN filters.")
parser.add_argument('-l2', '--l2-reg', type=float, default=1e-4, help="L2 regularization coefficient.")
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train.")
parser.add_argument('--batch-size', type=int, default=64, help="Batch size.")
parser.add_argument('--model-name', type=str, default='GAE_model', help="Name of the GAE model to be saved.")
parser.add_argument('--results_dir', type=str, default='./results/', help="Directory to store results.")
parser.add_argument('--train-list', type=str, default='train_domains.txt', help="List with train structure IDs.")
parser.add_argument('--test-list', type=str, default='test_domains.txt', help="List with test structure IDs.")
args = parser.parse_args()

# CUDA for PyTorch
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

# load entire list
domains = load_domain_list(path + 'cath-dataset-nonredundant-S40.list')

np.random.seed(1234)
np.random.shuffle(domains)

args.train_list = domains[:25000]
args.valid_list = domains[25000:30000]
args.test_list = domains[28000:]


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


if __name__ == "__main__":
    print (args.model_name)
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
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * args.batch_size, len(training_generator.dataset),
                                                                               100. * batch_idx / len(training_generator), loss))
        print('====> Epoch: {} Average train_loss: {:.4f}'.format(epoch, train_loss / len(training_generator)))
        sys.stdout.flush()
        train_loss_list.append(train_loss/len(training_generator))
        with torch.no_grad():
            valid_loss = 0.0
            for batch_idx, (A_batch, S_batch) in enumerate(validation_generator):
                A_batch = A_batch.to(device)
                S_batch = S_batch.to(device)

                loss = valid_step((A_batch, S_batch), (A_batch, S_batch))
                valid_loss += loss
        print('====> Epoch: {} Average valid_loss: {:.4f}'.format(epoch, valid_loss / len(validation_generator)))
        sys.stdout.flush()
        valid_loss_list.append(valid_loss/len(validation_generator))

    # plot lossess
    plot_losses(train_loss_list, valid_loss_list)

    # save model
    torch.save(gae.state_dict(), args.results_dir + args.model_name + '_model.pt')
