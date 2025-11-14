"""Utilities to assemble a basic alignment QUBO.

The formulation follows the description in README.md:

H_total = H_node + H_structure + H_constraint,

where both constraint components enforce *at-most-one* assignments for
entities on either side of the alignment.
"""

from collections import defaultdict
from typing import Dict, Tuple, Optional

import torch


def _add(Q: Dict[Tuple[int, int], float], u: int, v: int, value: float) -> None:
    """
    helper to add a coefficient to the QUBO dictionary.
    
    Parameters
    ----------
    Q : Dict[Tuple[int, int], float]
        The QUBO dictionary to modify.
    u : int
        The first variable index.
    v : int
        The second variable index.
    value : float
        The coefficient to add.
    
    Returns
    -------
    None
    """
    
    # accumulate coefficients while keeping matrix upper-triangular
    if u > v:  # enforce ordering
        u, v = v, u
    Q[(u, v)] = Q.get((u, v), 0.0) + float(value)  # sum existing coefficient


def _build_variable_index(similarity: torch.Tensor, threshold: Optional[float] = None):
    # assign binary variable ids to viable wiki/arxiv pairs
    num_wiki, num_arxiv = similarity.shape  # dimensions of similarity grid
    index = {}  # (wiki, arxiv) -> variable id
    reverse = {}  # variable id -> (wiki, arxiv)

    counter = 0  # running variable id
    for i in range(num_wiki):  # iterate wiki nodes
        for j in range(num_arxiv):  # iterate arxiv nodes
            score = float(similarity[i, j])  # convert tensor to python float
            if threshold is not None and score < threshold:  # drop weak matches
                continue
            index[(i, j)] = counter  # assign id
            reverse[counter] = (i, j)  # remember reverse mapping
            counter += 1  # bump id

    return index, reverse


def formulate(
    node_info,
    structural_info,
    *,
    node_weight: float = 1.0,
    structural_weight: float = 1.0,
    wiki_penalty: float = 2.0,
    arxiv_penalty: float = 2.0,
    similarity_threshold: Optional[float] = None,
):
    # translate similarity data into a raw QUBO dictionary
    similarity = node_info["similarity"]
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity)  # accept numpy arrays or lists

    var_index, reverse_index = _build_variable_index(similarity, similarity_threshold)  # variable ids
    Q = defaultdict(float)  # sparse qubo accumulator

    # linear rewards for matching similar entities. (H_node)
    for (i, j), var in var_index.items():
        score = float(similarity[i, j])  # similarity score
        _add(Q, var, var, -node_weight * score)  # negative reward encourages matches

    # quadratic rewards for matching consistent structures. (H_structure)
    structural_weights = structural_info.get("weights", {})
    for (i, j, a, b), weight in structural_weights.items():
        left = var_index.get((i, a))  # check variable exists
        right = var_index.get((j, b))  # check variable exists
        if left is None or right is None:  # skip missing nodes
            continue
        _add(Q, left, right, -structural_weight * float(weight))  # reward consistent structure

    # at-most-one penalties across wiki nodes (H_constraint).
    wiki_nodes = node_info["wiki_nodes"]
    arxiv_nodes = node_info["arxiv_nodes"]
    for i, _ in enumerate(wiki_nodes):
        active = [var_index[(i, a)] for a in range(len(arxiv_nodes)) if (i, a) in var_index]  # viable matches
        for idx, left in enumerate(active):  # pair every combination
            for right in active[idx + 1 :]:
                _add(Q, left, right, wiki_penalty)  # penalty discourages double matches

    # at-most-one penalties across arXiv nodes.
    for a, _ in enumerate(arxiv_nodes):
        active = [var_index[(i, a)] for i in range(len(wiki_nodes)) if (i, a) in var_index]  # viable matches
        for idx, left in enumerate(active):  # pair every combination
            for right in active[idx + 1 :]:
                _add(Q, left, right, arxiv_penalty)  # penalize duplicate pulls

    result = {
        "Q": dict(Q),
        "index": var_index,
        "reverse_index": reverse_index,
        "constants": 0.0,
    }
    return result


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:  # allow running as script
        sys.path.insert(0, str(repo_root))

    from src.qubo_alignment import weights

    parser = argparse.ArgumentParser(description="Build a trial alignment QUBO")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum node similarity required to create a binary variable.",
    )
    parser.add_argument(
        "--max-structural-pairs",
        type=int,
        default=None,
        help="Optional cap on the number of structural terms (for quick sanity checks).",
    )
    args = parser.parse_args()

    node_info = weights.load_node_similarity()  # gather node similarities
    structural_info = weights.load_structural_weights(
        node_info["wiki_lookup"], node_info["arxiv_lookup"]
    )  # gather relation weights

    if args.max_structural_pairs is not None:
        structural_weights = structural_info["weights"]  # full structural dict
        limited_items = list(structural_weights.items())[: args.max_structural_pairs]  # truncate for speed
        structural_info = {
            **structural_info,
            "weights": dict(limited_items),
        }

    result = formulate(
        node_info,
        structural_info,
        similarity_threshold=args.similarity_threshold,
    )

    num_variables = len(result["index"])  # variable count for quick sanity check
    num_terms = len(result["Q"])  # number of non-zero coefficients

    # report quick stats for inspection
    print(f"QUBO variables: {num_variables}")
    print(f"Non-zero Q entries: {num_terms}")

    if num_terms:
        sample = list(result["Q"].items())[:5]  # preview first few terms
        print("Sample Q entries:")
        for (u, v), coeff in sample:
            print(f"  ({u}, {v}) -> {coeff:.4f}")
