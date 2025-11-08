import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import InnerProductDecoder

class GraphAutoencoder(torch.nn.Module):
    """
    A minimal Graph Autoencoder (GAE) model for unsupervised link prediction.
    
    It uses a GCN-based encoder and a simple inner product decoder.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphAutoencoder, self).__init__()
        
        # --- The Encoder ---
        # GCNConv is a Graph Convolutional Network layer
        # It learns by aggregating information from a node's neighbors.
        # We stack two layers to get information from 2-hop neighbors.
        
        # First layer: input features -> hidden features
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Second layer: hidden features -> output embedding
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Light regularization to reduce collapse
        self.dropout = torch.nn.Dropout(p=0.2)
        
        # --- The Decoder ---
        # The InnerProductDecoder simply takes the embeddings (vectors)
        # of two nodes and computes their dot product.
        # A high dot product = a high probability of a link.
        self.decoder = InnerProductDecoder()

    def encode(self, x, edge_index):
        """
        Runs the node features (x) and graph structure (edge_index)
        through the GCN encoder to get the final node embeddings (Z).
        """
        # Pass through the first GCN layer, apply ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Pass through the second GCN layer to get the final embeddings
        # We don't apply an activation here; this is the final "latent space"
        z = self.conv2(x, edge_index)
        return z

    def decode(self, z, edge_index):
        """
        Given the node embeddings (Z), the decoder predicts the
        existence of links (an adjacency matrix).
        """
        return self.decoder(z, edge_index)

    def forward(self, x, edge_index):
        """
        The full forward pass: Encode -> Decode
        """
        # Get the node embeddings
        z = self.encode(x, edge_index)
        
        # Get the reconstructed adjacency matrix (link predictions)
        adj_pred = self.decode(z, edge_index)
        
        return adj_pred