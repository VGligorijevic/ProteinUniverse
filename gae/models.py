import torch
import torch.nn as nn
from .layers import MultiGraphConv, InnerProductDecoder


class GAE(nn.Module):
    """
    Graph Autoencoder with a stack of Graph Convolution Layers and ZZ^T decoder.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 bias=False,
                 adj_sq=False,
                 scale_identity=False,
                 device=None):
        super(GAE, self).__init__()
        self.filters = filters

        # Sequence embedding
        self.seq_embedd = nn.Sequential(nn.Linear(in_features, self.filters[0]), nn.ReLU(inplace=True))

        # Encoder
        """
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

        # Decoder
        self.cmap_decoder = InnerProductDecoder(in_features=sum(self.filters), out_features=out_features)
        self.seq_decoder = nn.Sequential(nn.Linear(sum(self.filters), out_features=in_features), nn.LogSoftmax(dim=-1))

    def forward(self, data):
        A, S = data
        S = self.seq_embedd(S)
        gcn_embedd = [nn.Sequential(*list(self.cmap_encoder.children())[:i+1])((A, S))[1] for i in range(0, len(self.filters))]
        x = torch.cat(gcn_embedd, -1)
        cmap_out = self.cmap_decoder(x)
        seq_out = self.seq_decoder(x)
        return cmap_out, seq_out


class Embedding(nn.Module):
    """
    Extracting embeddings from the GraphConv layers of a pre-trained model.
    """
    def __init__(self, original_model):
        super(Embedding, self).__init__()
        self.seq_embedd = original_model.seq_embedd
        self.cmap_encoder = original_model.cmap_encoder
        self.num = len(list(self.cmap_encoder))
        self.gcn_layers = [self.cmap_encoder[:i] for i in range(1, self.num + 1)]

    def forward(self, x):
        x = [self.gcn_layers[i]((x[0], self.seq_embedd(x[1])))[1] for i in range(0, self.num)]
        x = torch.cat(x, -1).sum(axis=1)
        return x
