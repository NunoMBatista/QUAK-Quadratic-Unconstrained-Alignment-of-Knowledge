import torch
import torch.nn.functional as F
import numpy as np
from rdflib import Graph, URIRef
from itertools import product

from src.config import (
    ENTITY_EMBEDDINGS_WIKI_PATH,
    ENTITY_EMBEDDINGS_ARXIV_PATH,
    REL_EMBEDDINGS_PATH,
    KG_WIKI_FINAL_PATH,
    KG_ARXIV_FINAL_PATH,
)


def _label_from_uri(value) -> str:
    # strip uri fragments to a human-friendly token
    text = str(value)  # keep original uri as string
    if "#" in text:  # drop fragment identifiers
        text = text.rsplit("#", 1)[-1]
    if "/" in text:  # drop trailing path segments
        text = text.rsplit("/", 1)[-1]
    return text.replace("_", " ")  # prefer spaces over underscores


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _load_embedding_bundle(path):
    # pull embeddings tensor and mapping dict from disk bundle
    bundle = torch.load(path, weights_only=False)  # stay compatible with older saves
    return bundle["embeddings"], bundle["map"]  # tensor plus uri index map


def _subset_embeddings(bundle_path, ttl_path):
    # filter embeddings to entities that appear in the pruned graph
    embeddings, node_map = _load_embedding_bundle(bundle_path)  # load tensor and lookup
    lookup = {}  # label -> embedding row id
    for uri, idx in node_map.items():  # traverse original mapping
        label = _normalize(_label_from_uri(uri))  # normalize to shared label space
        lookup[label] = idx  # remember index for quick access

    graph = Graph()  # temporary graph loader
    graph.parse(str(ttl_path), format="turtle")  # load ttl subset

    ordered_nodes = []  # keep unique uris in graph order
    seen = set()  # guard against duplicates
    for s, _, o in graph:  # scan triples
        for node in (s, o):  # subjects and objects both considered
            if isinstance(node, URIRef) and node not in seen:  # focus on entities only
                seen.add(node)  # mark as processed
                ordered_nodes.append(node)  # append to ordered list

    ordered_nodes.sort(key=lambda uri: _normalize(_label_from_uri(uri)))

    filtered_nodes = []  # nodes that have embeddings
    indices = []  # corresponding embedding rows
    for uri in ordered_nodes:  # maintain deterministic order
        label = _normalize(_label_from_uri(uri))  # match lookup key
        idx = lookup.get(label)  # align to embedding row
        if idx is None:  # skip nodes without vectors
            continue
        filtered_nodes.append(uri)  # keep uri for later reference
        indices.append(idx)  # store index for slicing

    subset = embeddings[indices]  # slice embeddings tensor
    return filtered_nodes, subset  # return uris plus vectors


def _load_relations(ttl_path, node_lookup):
    # convert graph triples into index-labelled edges
    graph = Graph()  # init temporary graph
    graph.parse(str(ttl_path), format="turtle")  # load ttl file

    edges = []  # collect (src, dst, relation)
    for s, p, o in graph:  # iterate triples
        if isinstance(s, URIRef) and isinstance(o, URIRef):  # only entity to entity
            ls = node_lookup.get(s)  # map subject to index
            lo = node_lookup.get(o)  # map object to index
            if ls is None or lo is None:  # ignore edges without embeddings
                continue
            label = _normalize(_label_from_uri(p))  # normalize predicate label
            edges.append((ls, lo, label))  # store structural relation
    return edges


def load_node_similarity():
    # build cosine similarity matrix between wiki and arxiv entities
    wiki_nodes, wiki_emb = _subset_embeddings(ENTITY_EMBEDDINGS_WIKI_PATH, KG_WIKI_FINAL_PATH)
    arxiv_nodes, arxiv_emb = _subset_embeddings(ENTITY_EMBEDDINGS_ARXIV_PATH, KG_ARXIV_FINAL_PATH)

    wiki_lookup = {uri: idx for idx, uri in enumerate(wiki_nodes)}  # uri -> row id
    arxiv_lookup = {uri: idx for idx, uri in enumerate(arxiv_nodes)}  # uri -> row id

    wiki_norm = F.normalize(wiki_emb, dim=1)  # unit vectors
    arxiv_norm = F.normalize(arxiv_emb, dim=1)  # unit vectors
    similarity = torch.matmul(wiki_norm, arxiv_norm.T)  # cosine similarity matrix

    return {
        "wiki_nodes": wiki_nodes,
        "arxiv_nodes": arxiv_nodes,
        "wiki_lookup": wiki_lookup,
        "arxiv_lookup": arxiv_lookup,
        "similarity": similarity,
    }


def load_structural_weights(wiki_lookup, arxiv_lookup):
    # compute relation-level cosine scores for structural consistency
    rel_embeddings = np.load(REL_EMBEDDINGS_PATH)  # load relation npz
    rel_vectors = {_normalize(name): rel_embeddings[name] for name in rel_embeddings.files}  # map labels

    def rel_sim(label_a, label_b):
        vec_a = rel_vectors.get(label_a)  # fetch relation vector a
        vec_b = rel_vectors.get(label_b)  # fetch relation vector b
        if vec_a is None or vec_b is None:  # skip unseen labels
            return None
        norm_a = np.linalg.norm(vec_a)  # magnitude of a
        norm_b = np.linalg.norm(vec_b)  # magnitude of b
        if norm_a == 0.0 or norm_b == 0.0:  # guard zero vectors
            return None
        vec_a = vec_a / norm_a  # normalize a
        vec_b = vec_b / norm_b  # normalize b
        return float(np.dot(vec_a, vec_b))  # cosine score

    wiki_edges = _load_relations(KG_WIKI_FINAL_PATH, wiki_lookup)  # wiki relation triples
    arxiv_edges = _load_relations(KG_ARXIV_FINAL_PATH, arxiv_lookup)  # arxiv relation triples

    structural = {}  # map of structural pair weights
    for (i, j, label_a), (a, b, label_b) in product(wiki_edges, arxiv_edges):  # cross every edge pair
        weight = rel_sim(label_a, label_b)  # relation similarity
        if weight is None:  # skip missing data
            continue
        structural[(i, j, a, b)] = weight  # record structural term

    return {
        "wiki_edges": wiki_edges,
        "arxiv_edges": arxiv_edges,
        "weights": structural,
    }
