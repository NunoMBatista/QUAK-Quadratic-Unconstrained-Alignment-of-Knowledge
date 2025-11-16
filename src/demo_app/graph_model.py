from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class GraphNode:
    """Simple node container used by the GUI builder."""

    id: str
    label: str
    description: str = ""
    position: Tuple[float, float] = (50.0, 50.0)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["position"] = list(self.position)
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "GraphNode":
        position = payload.get("position", [50.0, 50.0])
        if isinstance(position, tuple):
            coords = position
        else:
            coords = tuple(position)  # type: ignore[arg-type]
        return cls(
            id=str(payload["id"]),
            label=str(payload.get("label", payload["id"])),
            description=str(payload.get("description", "")),
            position=(float(coords[0]), float(coords[1])),
        )


@dataclass
class GraphEdge:
    """Relationship between two nodes."""

    source: str
    target: str
    relation: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "GraphEdge":
        return cls(
            source=str(payload["source"]),
            target=str(payload["target"]),
            relation=str(payload.get("relation", "related_to")),
        )


class GraphModel:
    """Mutable in-memory graph used by the handmade KG demo."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------
    def add_node(
        self,
        label: str,
        description: str = "",
        position: Optional[Tuple[float, float]] = None,
    ) -> GraphNode:
        node_id = f"n{self._next_id}"
        self._next_id += 1
        node = GraphNode(
            id=node_id,
            label=label.strip() or f"Node {node_id}",
            description=description.strip(),
            position=position or (50.0, 50.0),
        )
        self.nodes[node.id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        del self.nodes[node_id]
        self.edges = [edge for edge in self.edges if edge.source != node_id and edge.target != node_id]

    def move_node(self, node_id: str, position: Tuple[float, float]) -> None:
        node = self.nodes.get(node_id)
        if not node:
            return
        node.position = position

    # ------------------------------------------------------------------
    # Edge helpers
    # ------------------------------------------------------------------
    def add_edge(self, source: str, target: str, relation: str) -> Optional[GraphEdge]:
        if source not in self.nodes or target not in self.nodes:
            return None
        relation = relation.strip() or "related_to"
        edge = GraphEdge(source=source, target=target, relation=relation)
        self.edges.append(edge)
        return edge

    def remove_edge(self, index: int) -> None:
        if 0 <= index < len(self.edges):
            self.edges.pop(index)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "GraphModel":
        model = cls(name=str(payload.get("name", "Graph")))
        model.nodes = {node_data["id"]: GraphNode.from_dict(node_data) for node_data in payload.get("nodes", [])}
        model.edges = [GraphEdge.from_dict(edge) for edge in payload.get("edges", [])]
        model._next_id = int(payload.get("next_id", len(model.nodes) + 1))
        return model

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def list_nodes(self) -> List[GraphNode]:
        return list(self.nodes.values())

    def list_edges(self) -> List[GraphEdge]:
        return list(self.edges)

    def unique_relations(self) -> List[str]:
        seen = set()
        for edge in self.edges:
            seen.add(edge.relation)
        return sorted(seen)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self._next_id = 1