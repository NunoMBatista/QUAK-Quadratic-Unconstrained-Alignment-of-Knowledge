from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, cast

import dimod
import numpy as np
import torch
import torch.nn.functional as F
from neal import SimulatedAnnealingSampler
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, coalesce, to_undirected
from transformers import AutoModel, AutoTokenizer

from src.config import (
    ANNEALER_BETA_RANGE,
    ANNEALER_NUM_READS,
    ANNEALER_SEED,
    ANNEALER_MAX_STRUCTURAL_PAIRS,
    EMBEDDING_DIM,
    GAEA_MMD_WEIGHT,
    GAEA_STATS_WEIGHT,
    GAEA_MAX_ALIGN_SAMPLES,
    HIDDEN_DIM,
    LEARNING_RATE,
    QUBO_ARXIV_PENALTY,
    QUBO_NODE_WEIGHT,
    QUBO_STRUCTURE_WEIGHT,
    QUBO_WIKI_PENALTY,
    SCIBERT_MODEL_NAME,
    USE_GAE_FOR_ENTITY_EMBEDDINGS,
    USE_SCIBERT_FEATURES,
)
from src.embedding.generate_embeddings import (
    get_mean_pooled_embedding,
    train_gaea_joint,
)
from src.qubo_alignment import formulate

from .graph_model import GraphModel, GraphNode


class EmbeddingMode(Enum):
    SCIBERT_ONLY = "SciBERT embeddings"
    SCIBERT_PLUS_GAE = "SciBERT + GNN-GAEA"
    GAE_ONLY = "GNN-GAEA only"

    @property
    def uses_scibert(self) -> bool:
        return self in (EmbeddingMode.SCIBERT_ONLY, EmbeddingMode.SCIBERT_PLUS_GAE)

    @property
    def uses_gae(self) -> bool:
        return self in (EmbeddingMode.SCIBERT_PLUS_GAE, EmbeddingMode.GAE_ONLY)

    @classmethod
    def default(cls) -> "EmbeddingMode":
        if USE_GAE_FOR_ENTITY_EMBEDDINGS:
            return cls.SCIBERT_PLUS_GAE if USE_SCIBERT_FEATURES else cls.GAE_ONLY
        return cls.SCIBERT_ONLY

    @classmethod
    def from_value(cls, value: str) -> "EmbeddingMode":
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Unknown embedding mode: {value}")


@dataclass
class AlignmentRow:
    wiki: str
    arxiv: str
    similarity: float
    structure_score: Optional[float] = None
    total_score: Optional[float] = None

    def as_nn_tuple(self) -> Tuple[str, str, str]:
        return (self.wiki, self.arxiv, f"{self.similarity:.3f}")

    def as_qubo_tuple(self) -> Tuple[str, str, str, str, str]:
        structure_display = "-" if self.structure_score is None else f"{self.structure_score:.3f}"
        total_display = "-" if self.total_score is None else f"{self.total_score:.3f}"
        return (
            self.wiki,
            self.arxiv,
            f"{self.similarity:.3f}",
            structure_display,
            total_display,
        )


@dataclass
class UnalignedEntity:
    graph: str
    label: str
    best_similarity: float
    reason: str

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return (
            self.graph,
            self.label,
            f"{self.best_similarity:.3f}",
            self.reason,
        )


@dataclass
class PipelineResult:
    nn_alignments: List[AlignmentRow]
    qubo_alignments: List[AlignmentRow]
    qubo_energy: Optional[float]
    logs: List[str]
    unaligned_entities: List[UnalignedEntity] = field(default_factory=list)


class ManualPipelineRunner:
    """Utility that mirrors the embedding/alignment flow for handmade graphs."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
        self.scibert = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
        self.scibert.eval()
        self._relation_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        wiki_graph: GraphModel,
        arxiv_graph: GraphModel,
        *,
        embedding_mode: Optional[EmbeddingMode] = None,
        node_weight: Optional[float] = None,
        structure_weight: Optional[float] = None,
        nn_threshold: Optional[float] = None,
        qubo_threshold: Optional[float] = None,
    ) -> PipelineResult:
        mode = embedding_mode or EmbeddingMode.default()
        node_info, structural_info, logs = self.prepare_inputs(
            wiki_graph,
            arxiv_graph,
            embedding_mode=mode,
        )
        resolved_node_weight = node_weight if node_weight is not None else QUBO_NODE_WEIGHT
        resolved_structure_weight = (
            structure_weight if structure_weight is not None else QUBO_STRUCTURE_WEIGHT
        )
        return self.solve_from_inputs(
            node_info,
            structural_info,
            logs,
            node_weight=resolved_node_weight,
            structure_weight=resolved_structure_weight,
            nn_threshold=nn_threshold,
            qubo_threshold=qubo_threshold,
        )

    def prepare_inputs(
        self,
        wiki_graph: GraphModel,
        arxiv_graph: GraphModel,
        *,
        embedding_mode: Optional[EmbeddingMode] = None,
    ) -> Tuple[Dict[str, object], Dict[str, object], List[str]]:
        mode = embedding_mode or EmbeddingMode.default()
        logs: List[str] = []
        if not wiki_graph.nodes:
            raise ValueError("The Wiki graph has no nodes. Add at least one entity before running the pipeline.")
        if not arxiv_graph.nodes:
            raise ValueError("The arXiv graph has no nodes. Add at least one entity before running the pipeline.")

        logs.append(f"Embedding mode: {mode.value}")
        logs.append("Building PyG datasets from handmade graphs...")
        wiki_data, wiki_nodes, wiki_edges = self._build_pyg_dataset(wiki_graph, mode)
        arxiv_data, arxiv_nodes, arxiv_edges = self._build_pyg_dataset(arxiv_graph, mode)

        logs.append("Training / preparing entity embeddings...")
        wiki_embeddings, arxiv_embeddings = self._generate_entity_embeddings(
            wiki_data,
            arxiv_data,
            logs,
            mode,
        )

        logs.append("Computing cosine similarity between entity embeddings...")
        similarity = self._build_similarity_matrix(wiki_embeddings, arxiv_embeddings)

        logs.append("Building relation embeddings for structural weights...")
        structural_info = self._build_structural_weights(wiki_edges, arxiv_edges)

        node_info: Dict[str, object] = {
            "wiki_nodes": wiki_nodes,
            "arxiv_nodes": arxiv_nodes,
            "similarity": similarity,
            "embedding_mode": mode.value,
        }
        logs.append("Embedding artifacts prepared. Matrices ready for inspection.")
        return node_info, structural_info, logs

    def solve_from_inputs(
        self,
        node_info: Dict[str, object],
        structural_info: Dict[str, object],
        logs: Optional[List[str]] = None,
        *,
        node_weight: float = QUBO_NODE_WEIGHT,
        structure_weight: float = QUBO_STRUCTURE_WEIGHT,
        nn_threshold: Optional[float] = None,
        qubo_threshold: Optional[float] = None,
    ) -> PipelineResult:
        log_buffer = logs if logs is not None else []
        log_buffer.append("Solving nearest-neighbor baseline alignments...")
        nn_alignments = self._nearest_neighbor_alignments(node_info, similarity_threshold=nn_threshold)

        log_buffer.append("Formulating QUBO problem...")
        qubo_data = formulate.formulate(
            node_info,
            structural_info,
            node_weight=node_weight,
            structural_weight=structure_weight,
            wiki_penalty=QUBO_WIKI_PENALTY,
            arxiv_penalty=QUBO_ARXIV_PENALTY,
            similarity_threshold=qubo_threshold,
        )
        if not qubo_data["index"]:
            raise ValueError(
                "The handmade graphs did not produce any candidate alignment pairs. Add more nodes or ensure embeddings are not identical."
            )

        log_buffer.append("Running simulated annealing on the handmade QUBO...")
        sampleset = self._solve_qubo(qubo_data["Q"])
        record = sampleset.first  # type: ignore[assignment]
        sample = cast(Dict[int, int], getattr(record, "sample"))
        qubo_alignments = self._extract_alignments(
            sample,
            qubo_data["reverse_index"],
            node_info,
            qubo_data["components"],
            node_weight,
        )
        qubo_energy = float(getattr(record, "energy"))
        log_buffer.append(f"QUBO energy: {qubo_energy:.4f} | alignments: {len(qubo_alignments)}")

        wiki_best, arxiv_best = self._best_similarities(node_info)
        unaligned_entities = self._deduplicate_unaligned(
            self._collect_threshold_misses(
                node_info,
                nn_threshold,
                "Nearest Neighbor",
                wiki_best,
                arxiv_best,
            )
            + self._collect_threshold_misses(
                node_info,
                qubo_threshold,
                "QUBO",
                wiki_best,
                arxiv_best,
            )
            + self._collect_unmatched_nodes(
                node_info,
                nn_alignments,
                "Nearest Neighbor",
                wiki_best,
                arxiv_best,
                nn_threshold,
            )
            + self._collect_unmatched_nodes(
                node_info,
                qubo_alignments,
                "QUBO",
                wiki_best,
                arxiv_best,
                qubo_threshold,
            )
        )

        return PipelineResult(
            nn_alignments=nn_alignments,
            qubo_alignments=qubo_alignments,
            qubo_energy=qubo_energy,
            logs=log_buffer,
            unaligned_entities=unaligned_entities,
        )

    # ------------------------------------------------------------------
    # Dataset prep
    # ------------------------------------------------------------------
    def _build_pyg_dataset(
        self,
        graph: GraphModel,
        mode: EmbeddingMode,
    ) -> Tuple[Data, List[GraphNode], List[Tuple[int, int, str]]]:
        nodes = sorted(graph.list_nodes(), key=lambda item: item.label.lower())
        node_lookup = {node.id: idx for idx, node in enumerate(nodes)}
        num_nodes = len(nodes)

        edge_pairs: List[Tuple[int, int]] = []
        labeled_edges: List[Tuple[int, int, str]] = []
        for edge in graph.list_edges():
            src = node_lookup.get(edge.source)
            dst = node_lookup.get(edge.target)
            if src is None or dst is None:
                continue
            edge_pairs.append((src, dst))
            labeled_edges.append((src, dst, self._normalize(edge.relation)))

        if edge_pairs:
            raw_edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            raw_edge_index = torch.empty((2, 0), dtype=torch.long)

        pos_edge_index = coalesce(
            to_undirected(raw_edge_index, num_nodes=num_nodes),
            num_nodes=num_nodes,
        )
        mp_edge_index, _ = add_self_loops(pos_edge_index, num_nodes=num_nodes)

        if num_nodes == 0:
            x = torch.empty((0, 0), dtype=torch.float32)
        elif mode.uses_scibert:
            features: List[np.ndarray] = []
            for node in nodes:
                text = node.label
                if node.description:
                    text = f"{node.label}. {node.description}"
                embedding = get_mean_pooled_embedding(text, self.tokenizer, self.scibert)
                features.append(embedding.numpy())
            stacked = np.stack(features, axis=0).astype(np.float32, copy=False)
            x = torch.from_numpy(stacked)
        else:
            x = torch.eye(num_nodes, dtype=torch.float32)

        if x.numel() > 0:
            x = F.normalize(x, p=2, dim=1)

        data = Data(x=x, edge_index=mp_edge_index)
        data.train_pos_edge_index = pos_edge_index
        return data, nodes, labeled_edges

    # ------------------------------------------------------------------
    def _generate_entity_embeddings(
        self,
        wiki_data: Data,
        arxiv_data: Data,
        logs: List[str],
        mode: EmbeddingMode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if wiki_data.x is None or arxiv_data.x is None:
            raise ValueError("Node feature tensors are missing; unable to run embeddings.")

        use_gae = mode.uses_gae
        has_edges = (
            wiki_data.train_pos_edge_index.size(1) > 0
            and arxiv_data.train_pos_edge_index.size(1) > 0
        )

        if use_gae and not has_edges:
            logs.append("Insufficient edges for GAE training; falling back to raw features.")
            use_gae = False

        if use_gae:
            wiki_nodes_count = int(wiki_data.num_nodes or wiki_data.x.size(0))
            arxiv_nodes_count = int(arxiv_data.num_nodes or arxiv_data.x.size(0))
            total_nodes = max(1, wiki_nodes_count + arxiv_nodes_count)
            joint_model = train_gaea_joint(
                wiki_data,
                arxiv_data,
                in_channels=wiki_data.num_node_features,
                hidden_channels=HIDDEN_DIM,
                out_channels=EMBEDDING_DIM,
                epochs=200,
                lr=LEARNING_RATE,
                mmd_weight=GAEA_MMD_WEIGHT,
                stats_weight=GAEA_STATS_WEIGHT,
                max_samples=min(GAEA_MAX_ALIGN_SAMPLES, total_nodes),
                anchor_pairs=None,
                anchor_weight=0.0,
            )
            joint_model.eval()
            with torch.no_grad():
                wiki_embeddings = joint_model.encode(wiki_data.x, wiki_data.edge_index)
                arxiv_embeddings = joint_model.encode(arxiv_data.x, arxiv_data.edge_index)
        else:
            wiki_embeddings = wiki_data.x
            arxiv_embeddings = arxiv_data.x

        if use_gae and mode is EmbeddingMode.SCIBERT_PLUS_GAE:
            wiki_embeddings = torch.cat([wiki_data.x, wiki_embeddings], dim=1)
            arxiv_embeddings = torch.cat([arxiv_data.x, arxiv_embeddings], dim=1)

        return wiki_embeddings.cpu(), arxiv_embeddings.cpu()

    # ------------------------------------------------------------------
    def _build_similarity_matrix(
        self,
        wiki_embeddings: torch.Tensor,
        arxiv_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        wiki_norm = F.normalize(wiki_embeddings, dim=1)
        arxiv_norm = F.normalize(arxiv_embeddings, dim=1)
        return torch.matmul(wiki_norm, arxiv_norm.T)

    # ------------------------------------------------------------------
    def _build_structural_weights(
        self,
        wiki_edges: List[Tuple[int, int, str]],
        arxiv_edges: List[Tuple[int, int, str]],
    ) -> Dict[str, object]:
        weights: Dict[Tuple[int, int, int, int], float] = {}
        if not wiki_edges or not arxiv_edges:
            return {
                "wiki_edges": wiki_edges,
                "arxiv_edges": arxiv_edges,
                "weights": weights,
            }

        def rel_vector(label: str) -> Optional[torch.Tensor]:
            normalized = self._normalize(label)
            cached = self._relation_cache.get(normalized)
            if cached is not None:
                return cached
            embedding = get_mean_pooled_embedding(label.replace("_", " "), self.tokenizer, self.scibert)
            if embedding.norm(p=2) == 0:
                return None
            normalized_vec = F.normalize(embedding.unsqueeze(0), dim=1).squeeze(0)
            self._relation_cache[normalized] = normalized_vec
            return normalized_vec

        for (i, j, rel_a) in wiki_edges:
            vec_a = rel_vector(rel_a)
            if vec_a is None:
                continue
            for (a, b, rel_b) in arxiv_edges:
                vec_b = rel_vector(rel_b)
                if vec_b is None:
                    continue
                weight = float(torch.dot(vec_a, vec_b).item())
                weights[(i, j, a, b)] = weight
        if ANNEALER_MAX_STRUCTURAL_PAIRS is not None:
            limited = list(weights.items())[: ANNEALER_MAX_STRUCTURAL_PAIRS]
            weights = dict(limited)
        return {
            "wiki_edges": wiki_edges,
            "arxiv_edges": arxiv_edges,
            "weights": weights,
        }

    # ------------------------------------------------------------------
    def _nearest_neighbor_alignments(
        self,
        node_info: Dict[str, object],
        similarity_threshold: Optional[float] = None,
    ) -> List[AlignmentRow]:
        similarity = cast(torch.Tensor, node_info["similarity"])
        wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]

        num_wiki, num_arxiv = similarity.shape
        candidates: List[Tuple[float, int, int]] = []
        for i in range(num_wiki):
            for j in range(num_arxiv):
                candidates.append((float(similarity[i, j]), i, j))

        candidates.sort(key=lambda item: item[0], reverse=True)
        used_wiki: set[int] = set()
        used_arxiv: set[int] = set()
        alignments: List[AlignmentRow] = []
        for score, i, j in candidates:
            if i in used_wiki or j in used_arxiv:
                continue
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            used_wiki.add(i)
            used_arxiv.add(j)
            alignments.append(
                AlignmentRow(
                    wiki=wiki_nodes[i].label,
                    arxiv=arxiv_nodes[j].label,
                    similarity=score,
                )
            )
        return alignments

    # ------------------------------------------------------------------
    def _solve_qubo(self, qubo: Dict[Tuple[int, int], float]) -> dimod.SampleSet:
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        sampler = SimulatedAnnealingSampler()
        return sampler.sample(
            bqm,
            num_reads=ANNEALER_NUM_READS,
            beta_range=ANNEALER_BETA_RANGE,
            seed=ANNEALER_SEED,
        )

    # ------------------------------------------------------------------
    def _extract_alignments(
        self,
        sample: Dict[int, int],
        reverse_index: Dict[int, Tuple[int, int]],
        node_info: Dict[str, object],
        components: Dict[str, Dict[Tuple[int, int], float]],
        node_weight: float,
    ) -> List[AlignmentRow]:
        wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]
        similarity: torch.Tensor = cast(torch.Tensor, node_info["similarity"])  # type: ignore[assignment]
        structure_terms = components.get("structure", {})

        alignments: List[AlignmentRow] = []
        for var, value in sample.items():
            if not value:
                continue
            wiki_idx, arxiv_idx = reverse_index[var]
            sim_value = float(similarity[wiki_idx, arxiv_idx])
            structure_score = self._structure_support(var, sample, structure_terms)
            total_score = sim_value * node_weight + structure_score
            alignments.append(
                AlignmentRow(
                    wiki=wiki_nodes[wiki_idx].label,
                    arxiv=arxiv_nodes[arxiv_idx].label,
                    similarity=sim_value,
                    structure_score=structure_score,
                    total_score=total_score,
                )
            )
        alignments.sort(key=lambda item: item.similarity, reverse=True)
        return alignments

    # ------------------------------------------------------------------
    @staticmethod
    def _structure_support(
        var: int,
        sample: Dict[int, int],
        structure_terms: Dict[Tuple[int, int], float],
    ) -> float:
        support = 0.0
        for (left, right), coeff in structure_terms.items():
            if left == var and sample.get(right, 0):
                support += -float(coeff)
            elif right == var and sample.get(left, 0):
                support += -float(coeff)
        return support

    # ------------------------------------------------------------------
    def _best_similarities(
        self,
        node_info: Dict[str, object],
    ) -> Tuple[List[float], List[float]]:
        wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]
        similarity: torch.Tensor = cast(torch.Tensor, node_info["similarity"])  # type: ignore[assignment]

        if similarity.size(1) == 0:
            wiki_best = [0.0 for _ in wiki_nodes]
        else:
            wiki_best = [float(similarity[i, :].max().item()) for i in range(similarity.size(0))]

        if similarity.size(0) == 0:
            arxiv_best = [0.0 for _ in arxiv_nodes]
        else:
            arxiv_best = [float(similarity[:, j].max().item()) for j in range(similarity.size(1))]

        return wiki_best, arxiv_best

    def _collect_threshold_misses(
        self,
        node_info: Dict[str, object],
        similarity_threshold: Optional[float],
        solver_label: str,
        wiki_best: List[float],
        arxiv_best: List[float],
    ) -> List[UnalignedEntity]:
        if similarity_threshold is None:
            return []
        wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]

        unaligned: List[UnalignedEntity] = []

        for idx, node in enumerate(wiki_nodes):
            best = wiki_best[idx]
            if best < similarity_threshold:
                unaligned.append(
                    UnalignedEntity(
                        graph="Wiki",
                        label=node.label,
                        best_similarity=best,
                        reason=f"{solver_label} threshold {similarity_threshold:.2f}",
                    )
                )

        for idx, node in enumerate(arxiv_nodes):
            best = arxiv_best[idx]
            if best < similarity_threshold:
                unaligned.append(
                    UnalignedEntity(
                        graph="arXiv",
                        label=node.label,
                        best_similarity=best,
                        reason=f"{solver_label} threshold {similarity_threshold:.2f}",
                    )
                )

        return unaligned

    # ------------------------------------------------------------------
    def _collect_unmatched_nodes(
        self,
        node_info: Dict[str, object],
        alignments: Sequence[AlignmentRow],
        solver_label: str,
        wiki_best: List[float],
        arxiv_best: List[float],
        threshold: Optional[float],
    ) -> List[UnalignedEntity]:
        wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]

        matched_wiki = {row.wiki for row in alignments}
        matched_arxiv = {row.arxiv for row in alignments}

        unaligned: List[UnalignedEntity] = []
        for idx, node in enumerate(wiki_nodes):
            if node.label in matched_wiki:
                continue
            if threshold is not None and wiki_best[idx] < threshold:
                continue
            unaligned.append(
                UnalignedEntity(
                    graph="Wiki",
                    label=node.label,
                    best_similarity=wiki_best[idx],
                    reason=f"{solver_label} solver unused",
                )
            )

        for idx, node in enumerate(arxiv_nodes):
            if node.label in matched_arxiv:
                continue
            if threshold is not None and arxiv_best[idx] < threshold:
                continue
            unaligned.append(
                UnalignedEntity(
                    graph="arXiv",
                    label=node.label,
                    best_similarity=arxiv_best[idx],
                    reason=f"{solver_label} solver unused",
                )
            )

        return unaligned

    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate_unaligned(entries: List[UnalignedEntity]) -> List[UnalignedEntity]:
        seen = set()
        deduped: List[UnalignedEntity] = []
        for entry in entries:
            key = (entry.graph, entry.label, entry.reason)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())
