from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import PROJECT_ROOT
from .graph_model import GraphModel

HANDMADE_DIR = PROJECT_ROOT / "hand_made_kgs"
HANDMADE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIENCE_DIR = HANDMADE_DIR / "experiences"
EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_name(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return slug or "graph"


def list_saved_graphs() -> List[Path]:
    if not HANDMADE_DIR.exists():
        return []
    return sorted(HANDMADE_DIR.glob("*.json"))


def save_graph(model: GraphModel, filename: str | Path | None = None) -> Path:
    HANDMADE_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        name = f"{_sanitize_name(model.name)}.json"
        path = HANDMADE_DIR / name
    else:
        candidate = Path(filename)
        if candidate.suffix != ".json":
            candidate = candidate.with_suffix(".json")
        if not candidate.is_absolute():
            candidate = HANDMADE_DIR / candidate
        path = candidate
    with path.open("w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle, indent=2)
    return path


def load_graph(path: Path) -> GraphModel:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return GraphModel.from_dict(payload)


def load_graph_by_name(name: str) -> GraphModel | None:
    slug = _sanitize_name(name)
    path = HANDMADE_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return load_graph(path)


def save_experience(
    wiki_graph: GraphModel,
    arxiv_graph: GraphModel,
    node_info: Dict[str, Any],
    structural_info: Dict[str, Any],
    filename: str | Path | None = None,
    qubo_weights: Optional[Dict[str, float]] = None,
) -> Path:
    EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        path = EXPERIENCE_DIR / "experience.json"
    else:
        candidate = Path(filename)
        if candidate.suffix != ".json":
            candidate = candidate.with_suffix(".json")
        if not candidate.is_absolute():
            candidate = EXPERIENCE_DIR / candidate
        path = candidate

    wiki_nodes = node_info.get("wiki_nodes", [])
    arxiv_nodes = node_info.get("arxiv_nodes", [])
    similarity = node_info.get("similarity")
    similarity_matrix: List[List[float]] = []
    if similarity is not None:
        similarity_matrix = [[float(value) for value in row] for row in similarity.tolist()]

    structural_weights = structural_info.get("weights", {})
    weights_payload = [
        {
            "wiki_i": int(key[0]),
            "wiki_j": int(key[1]),
            "arxiv_a": int(key[2]),
            "arxiv_b": int(key[3]),
            "weight": float(value),
        }
        for key, value in structural_weights.items()
    ]

    payload = {
        "wiki_graph": wiki_graph.to_dict(),
        "arxiv_graph": arxiv_graph.to_dict(),
        "wiki_order": [node.id for node in wiki_nodes],
        "arxiv_order": [node.id for node in arxiv_nodes],
        "similarity": similarity_matrix,
        "structural_info": {
            "wiki_edges": structural_info.get("wiki_edges", []),
            "arxiv_edges": structural_info.get("arxiv_edges", []),
            "weights": weights_payload,
        },
    }

    if qubo_weights is not None:
        payload["qubo_weights"] = {
            "node_weight": float(qubo_weights.get("node_weight", 0.0)),
            "structure_weight": float(qubo_weights.get("structure_weight", 0.0)),
        }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_experience(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    wiki_graph = GraphModel.from_dict(payload["wiki_graph"])
    arxiv_graph = GraphModel.from_dict(payload["arxiv_graph"])
    wiki_order = payload.get("wiki_order", [])
    arxiv_order = payload.get("arxiv_order", [])
    similarity = payload.get("similarity", [])
    structural_info = payload.get("structural_info", {})
    qubo_weights = payload.get("qubo_weights")

    return {
        "wiki_graph": wiki_graph,
        "arxiv_graph": arxiv_graph,
        "wiki_order": wiki_order,
        "arxiv_order": arxiv_order,
        "similarity": similarity,
        "structural_info": structural_info,
        "qubo_weights": qubo_weights,
    }
