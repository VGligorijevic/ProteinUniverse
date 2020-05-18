import torch
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F
from torch.nn import Parameter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GraphConv(nn.Module):
    """
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 activation=None,
                 bnorm=False,
                 adj_sq=False,
                 scale_identity=False,
                 device=None):
        super(GraphConv, self).__init__()
        # self.fc = nn.Linear(in_features, out_features, bias=bias, init='xavier')
        self.weight = Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter('bias', None)

        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()

        self.activation = activation
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity
        self.device = device

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, data):
        A, X = data
        batch, N = A.shape[:2]
        if self.device is not None:
            I = torch.eye(N).unsqueeze(0).to(self.device)
        else:
            I = torch.eye(N).unsqueeze(0)

        if self.scale_identity:
            I = 2 * I  # increase weight of self connections

        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity

        # A_hat = A + I
        A_hat = A
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        A_hat = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        x = torch.bmm(A_hat, X)
        x = torch.matmul(x, self.weight)

        if self.bias is not None:
            x = x + self.bias

        if self.bnorm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.activation is not None:
            x = self.activation(x)

        return (A, x)


class MultiGraphConv(nn.Module):
    """
    Graph Convolution Layer according to https://arxiv.org/pdf/1907.05008.pdf
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 activation=None,
                 bnorm=False,
                 device=None):
        super(MultiGraphConv, self).__init__()
        self.device = device
        self.weight_0 = Parameter(torch.Tensor(in_features, out_features))
        self.weight_1 = Parameter(torch.Tensor(in_features, out_features))
        self.weight_2 = Parameter(torch.Tensor(in_features, out_features))
        self.weight = Parameter(torch.Tensor(3*out_features, out_features)).to(self.device)
        if bias:
            self.bias_0 = Parameter(torch.Tensor(out_features))
            self.bias_1 = Parameter(torch.Tensor(out_features))
            self.bias_2 = Parameter(torch.Tensor(out_features))
            self.bias = Parameter(torch.Tensor(out_features)).to(self.device)
        else:
            self.register_parameter('bias', None)

        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(3*out_features)
        self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_0)
        nn.init.xavier_uniform_(self.weight_1)
        nn.init.xavier_uniform_(self.weight_2)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias_0)
            nn.init.zeros_(self.bias_1)
            nn.init.zeros_(self.bias_2)
            nn.init.zeros_(self.bias)

    def forward(self, data):
        A, X = data
        batch, N = A.shape[:2]

        D_1 = (torch.sum(A, 1) + 1e-5) ** (-0.5)
        A_1 = D_1.view(batch, N, 1) * A * D_1.view(batch, 1, N)

        D_2 = (torch.sum(A, 1) + 1e-5) ** (-1.0)
        A_2 = D_2.view(batch, N, 1) * A

        x = [torch.bmm(_A, X) for _A in [A, A_1, A_2]]
        x = [torch.matmul(x[0], self.weight_0), torch.matmul(x[1], self.weight_1), torch.matmul(x[2], self.weight_2)]

        if self.bias is not None:
            x = [x[0] + self.bias_0, x[1] + self.bias_1, x[2] + self.bias_2]

        # concatenate
        x = torch.cat(x, -1)

        if self.bnorm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.activation is not None:
            x = self.activation(x)

        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        if self.activation is not None:
            x = self.activation(x)

        return (A, x)


class InnerProductDecoder(nn.Module):
    """
    The inner product decoder accoding to (T. Kipf and M. Welling, arXiv:1611.07308, 2016)
    """
    def __init__(self, in_features=128, out_features=64):
        super(InnerProductDecoder, self).__init__()
        self.sigmoid = nn.Sigmoid()

        # Linear Projections
        self.proj_P = nn.Linear(in_features, out_features, bias=False)
        self.proj_Q = nn.Linear(in_features, out_features, bias=True)

    def forward(self, z):
        P = self.proj_P(z)
        Q = self.proj_Q(z)
        adj = torch.bmm(P, Q.transpose(2, 1))
        adj = (adj + adj.transpose(2, 1))
        adj = self.sigmoid(adj)
        return adj
