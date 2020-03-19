import torch
import torch.nn as nn
from .layers import GraphConv, InnerProductDecoder


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

        # Encoder
        self.encoder = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                  out_features=f,
                                                  bnorm=True,
                                                  activation=nn.ReLU(inplace=True),
                                                  bias=bias,
                                                  adj_sq=adj_sq,
                                                  scale_identity=scale_identity,
                                                  device=device) for layer, f in enumerate(self.filters)]))

        # Decoder
        self.cmap_decoder = InnerProductDecoder(in_features=sum(self.filters), out_features=out_features)
        self.seq_decoder = nn.Linear(sum(self.filters), out_features=22)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        gcn_embedd = [nn.Sequential(*list(self.encoder.children())[:i+1])(data)[1] for i in range(0, len(self.filters))]
        # x = self.encoder(data)[1]
        x = torch.cat(gcn_embedd, -1)
        cmap_out = self.cmap_decoder(x)
        seq_out = self.log_softmax(self.seq_decoder(x))
        return cmap_out, seq_out


class Embedding(nn.Module):
    """
    Extracting embeddings from the GraphConv layers of a pre-trained model.
    """
    def __init__(self, original_model):
        super(Embedding, self).__init__()
        self.num = len(list(original_model.children())[0])
        self.gcn_layers = [nn.Sequential(*list(original_model.children())[0][:i]) for i in range(1, self.num+1)]

    def forward(self, x):
        x = [self.gcn_layers[i](x)[1].sum(axis=1) for i in range(0, self.num)]
        x = torch.cat(x, -1)
        return x
