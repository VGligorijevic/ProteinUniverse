import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import Parameter


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
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
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
