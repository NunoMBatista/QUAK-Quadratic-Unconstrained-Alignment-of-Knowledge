from IPython import embed
from sympy.physics.units import kg
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops

from transformers import AutoTokenizer, AutoModel
import numpy as np
from rdflib import Graph, URIRef
import os
from src.config import *
from pathlib import Path
# --- Import your model definition ---
from src.embedding.gae_model import GraphAutoencoder

# --- Configuration ---
# Part 1: SciBERT (for Relations)
SCIBERT_MODEL_NAME = 'allenai/scibert_scivocab_cased'

# Part 2: GAE (for Entities)

# GAE Model & Training parameters
EMBEDDING_DIM = 64 # Final vector size (e.g., 64)
HIDDEN_DIM = 128  # GCN hidden layer size
EPOCHS = 200      # Number of training loops
LEARNING_RATE = 0.01

# --- Part 1: SciBERT Relation Embeddings ---

def get_mean_pooled_embedding(text, tokenizer, model):
    """Helper to get a single vector for a string via mean pooling."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.numpy()

def generate_relation_embeddings():
    """Generates and saves SciBERT embeddings for relation labels."""
    print(f"\n--- Part 1: Generating Relation Embeddings (for H_structure) ---")
    print(f"Loading SciBERT model: {SCIBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
    model.eval()
    
    embeddings_dict = {}
    print("Generating embeddings for relations:")
    for label in RELATION_LABELS:
        print(f"  - {label}")
        vec = get_mean_pooled_embedding(label, tokenizer, model)
        embeddings_dict[label] = vec
        
    #np.save(REL_EMBEDDINGS_PATH, embeddings_dict)
    np.savez(REL_EMBEDDINGS_PATH, **embeddings_dict)
    
    print(f"Successfully saved relation embeddings to {REL_EMBEDDINGS_PATH}")

# --- Part 2: GAE Entity Embeddings ---

def _get_entity_label_from_uri(uri):
    """Helper to extract a readable label from a URI."""
    return uri.split('/')[-1].replace('_', ' ')

def load_pyg_data_from_ttl(ttl_path, use_scibert_features=True):
    """
    Loads an rdflib Graph from a .ttl file and converts it
    into a PyTorch Geometric Data object.

    If use_scibert_features is True, it will generate SciBERT embeddings
    as node features. Otherwise, it will use an identity matrix.
    """
    print(f"  Loading graph from {ttl_path}...")
    g = Graph()
    g.parse(ttl_path, format="turtle")

    triples = list(g)

    # 1. Create a mapping from URI -> integer index
    # We only map subjects and objects (entities), not predicates
    all_uris = set()
    for s, p, o in triples:
            if isinstance(s, URIRef): all_uris.add(s)
            if isinstance(o, URIRef): all_uris.add(o)

    # Sort for consistent ordering
    unique_uris = sorted(list(all_uris))
    node_to_index = {uri: i for i, uri in enumerate(unique_uris)}
    num_nodes = len(unique_uris)

    # 2. Create the edge_index tensor
    source_nodes = []
    target_nodes = []

    for s, p, o in triples:
        # Check if both subject and object are entities we mapped
        if s in node_to_index and o in node_to_index:
            s_idx = node_to_index[s]
            o_idx = node_to_index[o]
            source_nodes.append(s_idx)
            target_nodes.append(o_idx)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # --- FIX: Make the graph undirected and add self-loops ---
    # This is a crucial step for GCN-based models like GAE.
    # It ensures message passing is bidirectional and nodes consider their own features.
    edge_index, _ = add_self_loops(to_undirected(edge_index), num_nodes=num_nodes)
    # --- END OF FIX ---

    # 3. Create node features (x)
    if use_scibert_features:
        print("  Generating SciBERT features for nodes...")
        tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
        model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
        model.eval()

        # Generate embeddings for each URI label
        node_features = []
        for uri in unique_uris: # This order matches the node_to_index mapping
            label = _get_entity_label_from_uri(str(uri))
            embedding = get_mean_pooled_embedding(label, tokenizer, model)
            node_features.append(embedding)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        # The GAE's input dimension will now be the SciBERT embedding dimension
        # Make sure to update the 'in_channels' parameter when calling train_gae
    else:
        # We don't have initial features, so we use a simple
        # identity matrix. GCNs can learn from this.
        print("  Using identity matrix for node features.")
        x = torch.eye(num_nodes)

    # 4. Create the Data object
    data = Data(x=x, edge_index=edge_index)

    # We also return the map to know which vector belongs to which URI
    print(f"  Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges.")
    return data, node_to_index

def train_gae(data, in_channels, hidden_channels, out_channels, epochs, lr):
    """
    Trains a single GAE model in an unsupervised way (link prediction).
    """
    model = GraphAutoencoder(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Get the graph edges (positive examples)
    pos_edge_index = data.edge_index
    
    print(f"  Training GAE for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. Encode
        z = model.encode(data.x, pos_edge_index)
        
        # 2. Get negative examples (random non-edges)
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes)
        
        # 3. Decode for both positive and negative edges
        pos_logits = model.decode(z, pos_edge_index)
        neg_logits = model.decode(z, neg_edge_index)
        
        # 4. Calculate Loss
        # We create labels: 1s for positive, 0s for negative
        pos_labels = torch.ones(pos_logits.shape[0])
        neg_labels = torch.zeros(neg_logits.shape[0])
        labels = torch.cat([pos_labels, neg_labels])
        logits = torch.cat([pos_logits, neg_logits])
        
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # 5. Backpropagate
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
            
    return model

def generate_entity_embeddings():
    """
    Loads, trains, and saves GAE embeddings for both KGs.
    """
    print(f"\n--- Part 2: Generating Entity Embeddings (for H_node) ---")

    # --- WIKI ---
    print("\nProcessing Wiki KG:")
    # WE TRAIN ON THE UNPRUNED KG TO GET BETTER EMBEDDINGS
    wiki_data, wiki_map = load_pyg_data_from_ttl(KG_WIKI_UNPRUNED_PATH)

    # Train GAE_Wiki
    gae_wiki = train_gae(wiki_data,
                         in_channels=wiki_data.x.shape[1], # Use feature dimension
                         hidden_channels=HIDDEN_DIM,
                         out_channels=EMBEDDING_DIM,
                         epochs=EPOCHS,
                         lr=LEARNING_RATE)

    # Get final embeddings from the trained encoder
    gae_wiki.eval()
    with torch.no_grad():
        wiki_embeddings = gae_wiki.encode(wiki_data.x, wiki_data.edge_index)
    
    # Save the embeddings AND the map
    torch.save({'embeddings': wiki_embeddings, 'map': wiki_map}, ENTITY_EMBEDDINGS_WIKI_PATH)
    print(f"Saved Wiki entity embeddings to {ENTITY_EMBEDDINGS_WIKI_PATH}")

    # --- ARXIV ---
    print("\nProcessing ArXiv KG:")
    arxiv_data, arxiv_map = load_pyg_data_from_ttl(KG_ARXIV_UNPRUNED_PATH)

    # Train GAE_Arxiv (independently!)
    gae_arxiv = train_gae(arxiv_data,
                          in_channels=arxiv_data.x.shape[1], # Use feature dimension
                          hidden_channels=HIDDEN_DIM,
                          out_channels=EMBEDDING_DIM,
                          epochs=EPOCHS,
                          lr=LEARNING_RATE)

    # Get final embeddings
    gae_arxiv.eval()
    with torch.no_grad():
        arxiv_embeddings = gae_arxiv.encode(arxiv_data.x, arxiv_data.edge_index)
    
    # Save the embeddings AND the map
    torch.save({'embeddings': arxiv_embeddings, 'map': arxiv_map}, ENTITY_EMBEDDINGS_ARXIV_PATH)
    print(f"Saved ArXiv entity embeddings to {ENTITY_EMBEDDINGS_ARXIV_PATH}")


# --- Main execution ---
if __name__ == "__main__":
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(REL_EMBEDDINGS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ENTITY_EMBEDDINGS_WIKI_PATH), exist_ok=True)
    
    # 1. Generate the H_structure weights
    generate_relation_embeddings()
    
    # 2. Generate the H_node hints
    generate_entity_embeddings()
    
    print("\n--- All Embeddings Generated ---")