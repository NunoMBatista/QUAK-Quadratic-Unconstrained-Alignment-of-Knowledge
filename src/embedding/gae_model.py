import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import InnerProductDecoder


class GraphEncoder(torch.nn.Module):
    """Two-layer GCN encoder used by the graph autoencoders."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)


class GraphAutoencoder(torch.nn.Module):
    """
    A minimal Graph Autoencoder (GAE) model for unsupervised link prediction.
    The encoder is shared here so the same backbone can be reused by
    single-graph and joint (GAEA) training strategies.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, out_channels, dropout=dropout)
        self.decoder = InnerProductDecoder()

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)