from IPython import embed
from sympy.physics.units import kg
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops, coalesce

from transformers import AutoTokenizer, AutoModel
import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF
import os
from typing import List
from src.config import *
from pathlib import Path
from src.embedding.gae_model import GraphAutoencoder

# Part 1: SciBERT (for Relations)
SCIBERT_MODEL_NAME = 'allenai/scibert_scivocab_cased'

# --- Part 1: SciBERT Relation Embeddings ---

def get_mean_pooled_embedding(text, tokenizer, model):
    """helper to get a single vector for a string via mean pooling."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.detach().cpu()

def _relation_label_from_uri(value: URIRef) -> str:
    text = str(value)
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.replace("_", " ").strip()


def _collect_relation_labels() -> List[str]:
    candidates = [
        KG_WIKI_UNPRUNED_PATH,
        KG_ARXIV_UNPRUNED_PATH,
        KG_WIKI_FINAL_PATH,
        KG_ARXIV_FINAL_PATH,
    ]
    labels = set()
    for path in candidates:
        ttl_path = Path(path)
        if not ttl_path.exists():
            continue
        graph = Graph()
        graph.parse(str(ttl_path), format="turtle")
        for _, predicate, _ in graph:
            if predicate == RDF.type:
                continue
            if not isinstance(predicate, URIRef):
                continue
            label = _relation_label_from_uri(predicate)
            if label:
                labels.add(label)
    if not labels:
        labels.update(RELATION_LABELS)
    return sorted(labels)


def _relation_key(label: str) -> str:
    return "_".join(label.lower().split())


def generate_relation_embeddings():
    """generates and saves SciBERT embeddings for relation labels."""

    print(f"\n--- Part 1: Generating Relation Embeddings (for H_structure) ---")
    print(f"Loading SciBERT model: {SCIBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
    model.eval()

    relation_labels = _collect_relation_labels()
    print(f"Discovered {len(relation_labels)} relation labels.")

    embeddings_dict = {}
    for label in relation_labels:
        text_label = label.replace("_", " ")
        key = _relation_key(text_label)
        if key in embeddings_dict:
            continue
        print(f"  - {text_label}")
        vec = get_mean_pooled_embedding(text_label, tokenizer, model).numpy()
        embeddings_dict[key] = vec

    np.savez(REL_EMBEDDINGS_PATH, **embeddings_dict)

    print(f"Successfully saved relation embeddings to {REL_EMBEDDINGS_PATH}")

# --- Part 2: GAE Entity Embeddings ---

def _get_entity_label_from_uri(uri):
    """helper to extract a readable label from a URI."""
    return uri.split('/')[-1].replace('_', ' ')

def load_pyg_data_from_ttl(ttl_path, tokenizer=None, model=None, use_scibert_features=USE_SCIBERT_FEATURES):
    """
    loads an rdflib Graph from a .ttl file and converts it
    into a PyTorch Geometric Data object.

    if use_scibert_features is True, it will generate SciBERT embeddings
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

    # Original directed edges
    raw_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # Use undirected edges for reconstruction and GCN message passing
    undirected_edges = to_undirected(raw_edge_index, num_nodes=num_nodes)
    pos_edge_index = coalesce(undirected_edges, num_nodes=num_nodes)

    # Add self-loops only for message passing (not for reconstruction loss)
    mp_edge_index, _ = add_self_loops(pos_edge_index, num_nodes=num_nodes)

    # 3. Create node features (x)
    if use_scibert_features:
        print("  Generating SciBERT features for nodes...")
        if tokenizer is None or model is None:
            tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
            model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
            model.eval()

        # Generate embeddings for each URI label
        node_features = []
        for uri in unique_uris:  # This order matches the node_to_index mapping
            label = _get_entity_label_from_uri(str(uri))
            embedding = get_mean_pooled_embedding(label, tokenizer, model)
            node_features.append(embedding.numpy())

        x = torch.tensor(np.array(node_features), dtype=torch.float)
        # The GAE's input dimension will now be the SciBERT embedding dimension
        # Make sure to update the 'in_channels' parameter when calling train_gae
    else:
        # We don't have initial features, so we use a simple
        # identity matrix. GCNs can learn from this.
        print("  Using identity matrix for node features.")
        x = torch.eye(num_nodes)

    # Normalize node features to stabilize training
    x = F.normalize(x, p=2, dim=1)

    # 4. Create the Data object
    data = Data(x=x, edge_index=mp_edge_index)
    data.train_pos_edge_index = pos_edge_index

    # We also return the map to know which vector belongs to which URI
    print(f"  Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges.")
    return data, node_to_index


def _sample_embeddings_for_alignment(z, max_samples):
    """Randomly subsample embeddings to keep MMD computation tractable."""
    if max_samples is None or max_samples <= 0 or z.size(0) <= max_samples:
        return z
    idx = torch.randperm(z.size(0), device=z.device)[:max_samples]
    return z.index_select(0, idx)


def _gaussian_kernel(x, y, gamma):
    distances = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-gamma * distances)


def _kernel_mean(kmat, same_input=False):
    if same_input:
        n = kmat.size(0)
        if n <= 1:
            return kmat.new_zeros(())
        sum_ex_diag = kmat.sum() - kmat.diagonal().sum()
        return sum_ex_diag / (n * (n - 1))
    return kmat.mean()


def _mmd_loss(z_a, z_b, max_samples=GAEA_MAX_ALIGN_SAMPLES):
    if z_a.size(0) == 0 or z_b.size(0) == 0:
        return z_a.new_zeros(())

    a = _sample_embeddings_for_alignment(z_a, max_samples)
    b = _sample_embeddings_for_alignment(z_b, max_samples)

    with torch.no_grad():
        combined = torch.cat([a, b], dim=0)
        pairwise = torch.cdist(combined, combined, p=2)
        mask = ~torch.eye(pairwise.size(0), dtype=torch.bool, device=pairwise.device)
        off_diag = pairwise[mask]
        if off_diag.numel() == 0:
            bandwidth = 1.0
        else:
            median = torch.median(off_diag)
            bandwidth = median.item() if median.item() > 0 else 1.0
        gamma = 1.0 / (2.0 * bandwidth ** 2)

    k_xx = _gaussian_kernel(a, a, gamma)
    k_yy = _gaussian_kernel(b, b, gamma)
    k_xy = _gaussian_kernel(a, b, gamma)

    return _kernel_mean(k_xx, same_input=True) + _kernel_mean(k_yy, same_input=True) - 2.0 * _kernel_mean(k_xy)


def _moment_alignment_loss(z_a, z_b):
    if z_a.size(0) == 0 or z_b.size(0) == 0:
        return z_a.new_zeros(())
    mean_loss = F.mse_loss(z_a.mean(dim=0), z_b.mean(dim=0))
    std_a = torch.sqrt(z_a.var(dim=0, unbiased=False) + 1e-6)
    std_b = torch.sqrt(z_b.var(dim=0, unbiased=False) + 1e-6)
    std_loss = F.mse_loss(std_a, std_b)
    return mean_loss + std_loss


def train_gae(data, in_channels, hidden_channels, out_channels, epochs, lr):
    """
    Trains a single GAE model in an unsupervised way (link prediction).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAutoencoder(in_channels, hidden_channels, out_channels, dropout=GAE_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Move data to the same device (PyG returns a new Data object)
    data_device = data.to(device)

    # Use the stored positive edges without self-loops for reconstruction
    pos_edge_index = data_device.train_pos_edge_index
    
    print(f"  Training GAE for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. Encode
        z = model.encode(data_device.x, data_device.edge_index)
        
        # 2. Get negative examples (random non-edges)
        neg_edge_index = negative_sampling(
            pos_edge_index,
            num_nodes=data_device.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # 3. Decode for both positive and negative edges
        pos_logits = model.decode(z, pos_edge_index)
        neg_logits = model.decode(z, neg_edge_index)
        
        # 4. Calculate Loss
        # We create labels: 1s for positive, 0s for negative
        pos_labels = torch.ones(pos_logits.shape[0], device=device)
        neg_labels = torch.zeros(neg_logits.shape[0], device=device)
        labels = torch.cat([pos_labels, neg_labels])
        logits = torch.cat([pos_logits, neg_logits])
        
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # 5. Backpropagate
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            pos_prob = torch.sigmoid(pos_logits).mean().item()
            neg_prob = torch.sigmoid(neg_logits).mean().item()
            print(
                f"    Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, "
                f"mean pos prob: {pos_prob:.3f}, mean neg prob: {neg_prob:.3f}"
            )

    # Move model back to CPU for downstream usage
    model.cpu()
    return model


def train_gaea_joint(
    wiki_data,
    arxiv_data,
    in_channels,
    hidden_channels,
    out_channels,
    epochs,
    lr,
    mmd_weight=GAEA_MMD_WEIGHT,
    stats_weight=GAEA_STATS_WEIGHT,
    max_samples=GAEA_MAX_ALIGN_SAMPLES,
):
    """Train a shared GAE (GAEA) that aligns both graphs simultaneously."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAutoencoder(in_channels, hidden_channels, out_channels, dropout=GAE_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    wiki_device = wiki_data.to(device)
    arxiv_device = arxiv_data.to(device)

    wiki_pos_edges = wiki_device.train_pos_edge_index
    arxiv_pos_edges = arxiv_device.train_pos_edge_index

    print(f"  Training joint GAEA for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z_wiki = model.encode(wiki_device.x, wiki_device.edge_index)
        z_arxiv = model.encode(arxiv_device.x, arxiv_device.edge_index)

        wiki_neg_edges = negative_sampling(
            wiki_pos_edges,
            num_nodes=wiki_device.num_nodes,
            num_neg_samples=wiki_pos_edges.size(1),
        ).to(device)
        arxiv_neg_edges = negative_sampling(
            arxiv_pos_edges,
            num_nodes=arxiv_device.num_nodes,
            num_neg_samples=arxiv_pos_edges.size(1),
        ).to(device)

        wiki_pos_logits = model.decode(z_wiki, wiki_pos_edges)
        wiki_neg_logits = model.decode(z_wiki, wiki_neg_edges)
        wiki_labels = torch.cat([
            torch.ones_like(wiki_pos_logits),
            torch.zeros_like(wiki_neg_logits),
        ])
        wiki_logits = torch.cat([wiki_pos_logits, wiki_neg_logits])
        wiki_loss = F.binary_cross_entropy_with_logits(wiki_logits, wiki_labels)

        arxiv_pos_logits = model.decode(z_arxiv, arxiv_pos_edges)
        arxiv_neg_logits = model.decode(z_arxiv, arxiv_neg_edges)
        arxiv_labels = torch.cat([
            torch.ones_like(arxiv_pos_logits),
            torch.zeros_like(arxiv_neg_logits),
        ])
        arxiv_logits = torch.cat([arxiv_pos_logits, arxiv_neg_logits])
        arxiv_loss = F.binary_cross_entropy_with_logits(arxiv_logits, arxiv_labels)

        alignment_loss = _mmd_loss(z_wiki, z_arxiv, max_samples=max_samples)
        stats_loss = _moment_alignment_loss(z_wiki, z_arxiv)

        loss = wiki_loss + arxiv_loss + mmd_weight * alignment_loss + stats_weight * stats_loss

        if torch.isnan(loss):
            raise ValueError("Encountered NaN while training the joint GAEA model.")

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                wiki_pos_prob = torch.sigmoid(wiki_pos_logits.detach()).mean().item()
                wiki_neg_prob = torch.sigmoid(wiki_neg_logits.detach()).mean().item()
                arxiv_pos_prob = torch.sigmoid(arxiv_pos_logits.detach()).mean().item()
                arxiv_neg_prob = torch.sigmoid(arxiv_neg_logits.detach()).mean().item()
            print(
                f"    Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, "
                f"Wiki recon: {wiki_loss.item():.4f}, ArXiv recon: {arxiv_loss.item():.4f}, "
                f"MMD: {alignment_loss.item():.4f}, Stats: {stats_loss.item():.4f}, "
                f"Wiki pos/neg: {wiki_pos_prob:.3f}/{wiki_neg_prob:.3f}, "
                f"ArXiv pos/neg: {arxiv_pos_prob:.3f}/{arxiv_neg_prob:.3f}"
            )

    model.cpu()
    return model

def generate_entity_embeddings():
    """Loads, trains, and saves entity embeddings for both KGs."""

    print(f"\n--- Part 2: Generating Entity Embeddings (for H_node) ---")

    # load SciBERT once and reuse for both graphs
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
    model.eval()

    print("\nProcessing Wiki KG:")
    wiki_data, wiki_map = load_pyg_data_from_ttl(
        KG_WIKI_UNPRUNED_PATH,
        tokenizer=tokenizer,
        model=model,
        use_scibert_features=USE_SCIBERT_FEATURES
    )

    if wiki_data.x is None:
        raise ValueError("Wiki graph is missing node features after preprocessing.")

    print("\nProcessing ArXiv KG:")
    arxiv_data, arxiv_map = load_pyg_data_from_ttl(
        KG_ARXIV_UNPRUNED_PATH,
        tokenizer=tokenizer,
        model=model,
        use_scibert_features=USE_SCIBERT_FEATURES
    )

    if arxiv_data.x is None:
        raise ValueError("ArXiv graph is missing node features after preprocessing.")

    if wiki_data.num_node_features != arxiv_data.num_node_features:
        raise ValueError(
            "Wiki and ArXiv graphs must share the same feature dimension for joint training."
        )

    def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return tensor
        return tensor.detach().clone().cpu()

    if USE_GAE_FOR_ENTITY_EMBEDDINGS:
        joint_model = train_gaea_joint(
            wiki_data,
            arxiv_data,
            in_channels=wiki_data.num_node_features,
            hidden_channels=HIDDEN_DIM,
            out_channels=EMBEDDING_DIM,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
        )

        joint_model.eval()
        with torch.no_grad():
            wiki_embeddings = joint_model.encode(wiki_data.x, wiki_data.edge_index)
            arxiv_embeddings = joint_model.encode(arxiv_data.x, arxiv_data.edge_index)

        wiki_final = wiki_embeddings
        arxiv_final = arxiv_embeddings
        # Ensure tensors are on the same device before concatenation to avoid
        # CPU/CUDA mismatch errors (torch.cat requires same device for all inputs).
        if USE_SCIBERT_FEATURES and wiki_data.x is not None:
            # Move node features to the embeddings device if needed
            try:
                emb_dev = wiki_embeddings.device
            except Exception:
                emb_dev = torch.device("cpu")
            wiki_x = wiki_data.x
            if getattr(wiki_x, "device", None) != emb_dev:
                wiki_x = wiki_x.to(emb_dev)
            wiki_final = torch.cat([wiki_x, wiki_embeddings], dim=1)

        if USE_SCIBERT_FEATURES and arxiv_data.x is not None:
            try:
                emb_dev = arxiv_embeddings.device
            except Exception:
                emb_dev = torch.device("cpu")
            arxiv_x = arxiv_data.x
            if getattr(arxiv_x, "device", None) != emb_dev:
                arxiv_x = arxiv_x.to(emb_dev)
            arxiv_final = torch.cat([arxiv_x, arxiv_embeddings], dim=1)

        wiki_final = _cpu_clone(wiki_final)
        arxiv_final = _cpu_clone(arxiv_final)
    else:
        print("\nSkipping GAE training; exporting raw node features as embeddings.")
        wiki_final = _cpu_clone(wiki_data.x)
        arxiv_final = _cpu_clone(arxiv_data.x)

    model_name = "GAEA" if USE_GAE_FOR_ENTITY_EMBEDDINGS else (
        "SciBERT" if USE_SCIBERT_FEATURES else "IdentityFeatures"
    )

    metadata_base = {
        "model": model_name,
        "use_gae": USE_GAE_FOR_ENTITY_EMBEDDINGS,
        "use_scibert_features": USE_SCIBERT_FEATURES,
    }

    if USE_GAE_FOR_ENTITY_EMBEDDINGS:
        metadata_base.update({
            "hidden_dim": HIDDEN_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "mmd_weight": GAEA_MMD_WEIGHT,
            "stats_weight": GAEA_STATS_WEIGHT,
            "max_align_samples": GAEA_MAX_ALIGN_SAMPLES,
        })

    torch.save(
        {
            "embeddings": wiki_final,
            "map": wiki_map,
            "metadata": {**metadata_base, "graph": "wiki"},
        },
        ENTITY_EMBEDDINGS_WIKI_PATH,
    )
    print(f"Saved Wiki entity embeddings to {ENTITY_EMBEDDINGS_WIKI_PATH}")

    torch.save(
        {
            "embeddings": arxiv_final,
            "map": arxiv_map,
            "metadata": {**metadata_base, "graph": "arxiv"},
        },
        ENTITY_EMBEDDINGS_ARXIV_PATH,
    )
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