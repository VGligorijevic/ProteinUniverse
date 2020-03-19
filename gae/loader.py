import torch
import numpy as np
from torch.utils import data
from Bio import SeqIO

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

path = '/mnt/ceph/users/vgligorijevic/ProteinDesign/CATH/'


def load_fasta(filename):
    """ Loads fasta file and returns a dictionary of sequences """
    domain2seq = {}
    for entry in SeqIO.parse(open(filename, 'r'), 'fasta'):
        seq = str(entry.seq)
        entry = str(entry.id)
        entry = entry.split('|')[-1]
        entry = entry.split('/')[0]
        domain2seq[entry] = seq
    return domain2seq

domain2seqres = load_fasta(path + 'cath-dataset-nonredundant-S40.fa')


def load_domain_list(filename):
    """ Load list of CATH domain names """
    l = []
    fRead = open(filename, 'r')
    for line in fRead:
        l.append(line.strip())
    fRead.close()
    return l


def seq2onehot(seq):
    """ Create 21-dim 1-hot embedding """
    chars = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', '-']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.eye(vocab_size)
    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def onehot2seq(S):
    chars = np.asarray(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', '-'])
    rind = np.argmax(np.exp(S[0]), 1)
    seq = "".join(list(chars[rind]))
    return seq


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

    # padd
    for i in range(len(batch)):
        A_padded[i][:lengths[i], :][:, :lengths[i]] = batch[i][0]
        S_padded[i][:lengths[i], :] = batch[i][1]

    return (A_padded, S_padded)


class Dataset(data.Dataset):
    """ Characterizes a dataset for PyTorch """
    def __init__(self, domain_IDs):
        'Initialization'
        self.domain_IDs = domain_IDs

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

        # Create contact maps (10A cutoff)
        A[A <= 10.0] = 1.0
        A[A > 10.0] = 0.0
        A = A - torch.diag(torch.diagonal(A))
        A = A + torch.eye(A.shape[1])

        return (A, S)


if __name__ == "__main__":
    domains = load_domain_list(path + 'cath-dataset-nonredundant-S40.list')
    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'collate_fn': collate_padd}
    training_dataset = Dataset(domains[:1000])
    training_generator = data.DataLoader(training_dataset, **params)

    # for epoch in range(10):
    for batch in training_generator:
        print (torch.diagonal(batch[0][0]))
