import torch
import numpy as np
from rdflib import Graph, URIRef
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import to_undirected, coalesce, negative_sampling
from pathlib import Path
from src.config import *

# --- Configuration ---
# Set these paths to match your project's config

# 1. Relation Embeddings (H_structure)

# 2. Entity Embeddings (H_node)
# We will validate the Wiki embeddings, since they trained correctly
ENTITY_EMBEDDINGS_PATH = ENTITY_EMBEDDINGS_WIKI_PATH
#KG_GROUND_TRUTH_PATH = KG_DIR / 'kg_wiki_final.ttl'
KG_GROUND_TRUTH_PATH = KG_WIKI_UNPRUNED_PATH

# --------------------------

def clean_uri(uri):
    """Helper to make URIs readable."""
    return str(uri).split('/')[-1].split('#')[-1]

def validate_relation_embeddings():
    """
    Loads and prints the contents of the .npy file.
    """
    print(f"--- 1. Validating Relation Embeddings (H_structure) ---")
    print(f"Loading {REL_EMBEDDINGS_PATH}...")
    
    try:
        # We must set allow_pickle=True because it's a dict of numpy arrays
        #rel_embeds = np.load(REL_EMBEDDINGS_PATH, allow_pickle=True).item()
        rel_embeds = np.load(REL_EMBEDDINGS_PATH)
    except Exception as e:
        print(f"ERROR: Could not load relation embeddings. {e}")
        return

    print("Successfully loaded. Contents:")
    for label in rel_embeds.files:
        vector = rel_embeds[label]
        print(f"  - Label: '{label}'")
    print("---------------------------------------------------\n")


def validate_entity_embeddings():
    """
    Loads the trained GAE embeddings, runs the decoder,
    and compares predictions to the ground-truth graph.
    """
    print(f"--- 2. Validating Entity Embeddings (H_node) ---")
    print(f"Loading {ENTITY_EMBEDDINGS_PATH}...")
    
    try:
        # The .pt file is a dictionary with embeddings and the map
        # --- THIS IS THE FIX ---
        # We must set weights_only=False to allow loading the
        # rdflib.term.URIRef objects stored in the 'map' dictionary.
        data = torch.load(ENTITY_EMBEDDINGS_PATH, weights_only=False)
        # --- END OF FIX ---
        
        embeddings = data['embeddings']
        node_map = data['map'] # {URI: index}
    except Exception as e:
        print(f"ERROR: Could not load entity embeddings. {e}")
        return

    # Create a reverse map to get readable names from indices
    # {index: URI_string}
    index_to_node = {idx: clean_uri(uri) for uri, idx in node_map.items()}
    num_nodes = len(index_to_node)
    
    print(f"Loaded {num_nodes} node embeddings (vector shape: {embeddings.shape[1]}).")
    
    device = embeddings.device

    # --- Build edge indices from the ground-truth graph ---
    print(f"\nLoading ground truth graph: {KG_GROUND_TRUTH_PATH}")
    g = Graph()
    g.parse(str(KG_GROUND_TRUTH_PATH), format="turtle")

    sources = []
    targets = []
    for s, _, o in g:
        if (s in node_map) and (o in node_map):
            s_idx = node_map[s]
            o_idx = node_map[o]
            if s_idx == o_idx:
                continue  # skip self-loops for evaluation
            sources.append(s_idx)
            targets.append(o_idx)

    if not sources:
        print("No evaluateable links found in the ground-truth graph.")
        return

    raw_edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
    pos_edge_index = coalesce(
        to_undirected(raw_edge_index, num_nodes=num_nodes),
        num_nodes=num_nodes
    )

    # --- Run the Decoder in the same regime as training ---
    decoder = InnerProductDecoder().to(device)

    pos_logits = decoder(embeddings, pos_edge_index)
    pos_probs = torch.sigmoid(pos_logits)

    neg_edge_index = negative_sampling(
        pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    ).to(device)
    neg_logits = decoder(embeddings, neg_edge_index)
    neg_probs = torch.sigmoid(neg_logits)

    print(f"Found {pos_edge_index.size(1)} positive evaluation edges.")
    print("\n--- Decoder Output Analysis (edge-level) ---")
    print(f"Positive mean: {pos_probs.mean().item():.4f} ± {pos_probs.std().item():.4f}")
    print(f"Negative mean: {neg_probs.mean().item():.4f} ± {neg_probs.std().item():.4f}")

    # Display top/bottom positives
    topk = torch.topk(pos_probs, k=min(10, pos_probs.numel()))
    print("\nTop positive edges:")
    for prob, idx in zip(topk.values.cpu().tolist(), topk.indices.cpu().tolist()):
        u = pos_edge_index[0, idx].item()
        v = pos_edge_index[1, idx].item()
        print(f"  - ({index_to_node[u]}, {index_to_node[v]}): {prob:.4f}")

    bottomk = torch.topk(-pos_probs, k=min(10, pos_probs.numel()))
    print("\nWeakest positive edges:")
    for neg_prob, idx in zip(bottomk.values.cpu().tolist(), bottomk.indices.cpu().tolist()):
        prob = -neg_prob
        u = pos_edge_index[0, idx].item()
        v = pos_edge_index[1, idx].item()
        print(f"  - ({index_to_node[u]}, {index_to_node[v]}): {prob:.4f}")

    # Sample a handful of negatives for inspection
    print("\nSampled negative edges:")
    sample_size = min(10, neg_probs.numel())
    perm = torch.randperm(neg_probs.numel(), device=device)[:sample_size].cpu().tolist()
    for idx in perm:
        u = neg_edge_index[0, idx].item()
        v = neg_edge_index[1, idx].item()
        prob = neg_probs[idx].item()
        print(f"  - ({index_to_node[u]}, {index_to_node[v]}): {prob:.4f}")

    print("---------------------------------------------------\n")


if __name__ == "__main__":
    # 1. Validate the .npy file
    validate_relation_embeddings()
    
    # 2. Validate the .pt file
    validate_entity_embeddings()