import torch
import numpy as np
from rdflib import Graph, URIRef
from torch_geometric.nn.models import InnerProductDecoder
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
    
    # --- Run the Decoder ---
    # We will get the "reconstructed adjacency matrix"
    # This matrix contains the decoder's prediction for *every possible pair*
    
    # 1. Initialize the same decoder used in the GAE
    decoder = InnerProductDecoder()
    
    # 2. Get the full matrix of scores (logits)
    # This is (Z @ Z.T)
    # full_adj_logits = decoder(embeddings) CALLING THE DECODER DIRECTLY DOESN'T WORK BECAUSE IT EXPECTS EDGE INDEX
    full_adj_logits = torch.matmul(embeddings, embeddings.t())

    
    # 3. Apply sigmoid to get probabilities (0.0 to 1.0)
    full_adj_probs = torch.sigmoid(full_adj_logits)

    # --- Compare to Ground Truth ---
    print(f"\nLoading ground truth graph: {KG_GROUND_TRUTH_PATH}")
    g = Graph()
    g.parse(str(KG_GROUND_TRUTH_PATH), format="turtle")
    
    # Find all "true" links in the graph
    true_links = set()
    for s, p, o in g:
        # We only care about links between entities (not 'rdf:type' etc.)
        if (s in node_map) and (o in node_map):
            s_idx = node_map[s]
            o_idx = node_map[o]
            # Add links in both directions for an undirected graph
            true_links.add(tuple(sorted((s_idx, o_idx))))

    print(f"Found {len(true_links)} true links in the graph.")

    # --- Print Validation Results ---
    print("\n--- Decoder Output Analysis ---")
    
    print("\nPredictions for TRUE links (scores should be HIGH, close to 1.0):")
    if not true_links:
        print("  (No true links found to validate)")
    for s_idx, o_idx in true_links:
        score = full_adj_probs[s_idx, o_idx].item()
        s_name = index_to_node[s_idx]
        o_name = index_to_node[o_idx]
        print(f"  - ({s_name}, {o_name}): {score:.4f}")

    print("\nPredictions for FALSE links (scores should be LOW, close to 0.0):")
    checked = 0
    # Check a few random pairs that are NOT in true_links
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Only check upper triangle
            if tuple(sorted((i, j))) not in true_links:
                if checked >= 5: # Limit to 5 examples
                    break
                
                score = full_adj_probs[i, j].item()
                i_name = index_to_node[i]
                j_name = index_to_node[j]
                print(f"  - ({i_name}, {j_name}): {score:.4f}")
                checked += 1
    
    if checked == 0:
        print("  (No false links found to validate - graph might be fully connected)")
        
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    # 1. Validate the .npy file
    validate_relation_embeddings()
    
    # 2. Validate the .pt file
    validate_entity_embeddings()