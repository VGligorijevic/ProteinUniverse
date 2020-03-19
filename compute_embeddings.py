import torch

import pickle
import argparse
import numpy as np
from gae.models import GAE
from gae.models import Embedding
from gae.loader import load_domain_list
from gae.loader import load_fasta, seq2onehot


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='PyTorch GAE Training.')
parser.add_argument('-dims', '--filter-dims', type=int, default=[64, 64, 64, 64, 64], nargs='+', help="Dimensions of GCN filters.")
parser.add_argument('--model-name', type=str, default='GAE_model', help="Name of the GAE model to be loaded.")
parser.add_argument('-o', '--out-pckl', type=str, default='cath_embeddings', help="Name of pickle file to store results.")
args = parser.parse_args()

# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = '/mnt/ceph/users/vgligorijevic/ProteinDesign/CATH/'
domain2seqres = load_fasta(path + 'cath-dataset-nonredundant-S40.fa')

# load entire list
domains = load_domain_list(path + 'cath-dataset-nonredundant-S40.list')
np.random.seed(1234)
np.random.shuffle(domains)
domains = domains[:3000]

# load a pre-trained GAE model
gae = GAE(in_features=22, out_features=args.filter_dims[-1], filters=args.filter_dims, device=device)
gae.load_state_dict(torch.load(args.model_name))
gae.to(device)
gae.eval()

# model for extracting features
F = Embedding(gae)

# test model
Feat = np.zeros((len(domains), sum(args.filter_dims)))
for i, domain in enumerate(domains):
    A_example = torch.load('/mnt/ceph/users/vgligorijevic/ProteinDesign/CATH/cath-nr-S40_tensors/' + domain + '.pt')
    A_example[A_example <= 10] = 1.0
    A_example[A_example > 10] = 0.0
    A_example = A_example - torch.diag(torch.diagonal(A_example))
    A_example = A_example + torch.eye(A_example.shape[1])
    A_example = A_example.view(-1, *A_example.numpy().shape)
    A_example = A_example.to(device)

    S_example = seq2onehot(domain2seqres[domain])
    S_example = S_example.reshape(1, *S_example.shape)
    S_example = torch.from_numpy(S_example).float()
    S_example = S_example.to(device)

    x = F((A_example, S_example))[0]
    Feat[i] = x.cpu().detach().numpy()

# save features in a file
pickle.dump({'Feat': Feat, 'domains': domains}, open(args.out_pckl, 'wb'))
