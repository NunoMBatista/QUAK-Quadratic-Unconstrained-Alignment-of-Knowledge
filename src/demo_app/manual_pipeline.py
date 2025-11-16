from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import dimod
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


@dataclass
class AlignmentRow:
    wiki: str
    arxiv: str
    similarity: float

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.wiki, self.arxiv, f"{self.similarity:.3f}")


@dataclass
class PipelineResult:
    nn_alignments: List[AlignmentRow]
    qubo_alignments: List[AlignmentRow]
    qubo_energy: Optional[float]
    logs: List[str]


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
    def run(self, wiki_graph: GraphModel, arxiv_graph: GraphModel) -> PipelineResult:
        node_info, structural_info, logs = self.prepare_inputs(wiki_graph, arxiv_graph)
        return self.solve_from_inputs(node_info, structural_info, logs)

    def prepare_inputs(
        self, wiki_graph: GraphModel, arxiv_graph: GraphModel
    ) -> Tuple[Dict[str, object], Dict[str, object], List[str]]:
        logs: List[str] = []
        if not wiki_graph.nodes:
            raise ValueError("The Wiki graph has no nodes. Add at least one entity before running the pipeline.")
        if not arxiv_graph.nodes:
            raise ValueError("The arXiv graph has no nodes. Add at least one entity before running the pipeline.")

        logs.append("Building PyG datasets from handmade graphs...")
        wiki_data, wiki_nodes, wiki_edges = self._build_pyg_dataset(wiki_graph)
        arxiv_data, arxiv_nodes, arxiv_edges = self._build_pyg_dataset(arxiv_graph)

        logs.append("Training / preparing entity embeddings...")
        wiki_embeddings, arxiv_embeddings = self._generate_entity_embeddings(
            wiki_data, arxiv_data, logs
        )

        logs.append("Computing cosine similarity between entity embeddings...")
        similarity = self._build_similarity_matrix(wiki_embeddings, arxiv_embeddings)

        logs.append("Building relation embeddings for structural weights...")
        structural_info = self._build_structural_weights(wiki_edges, arxiv_edges)

        node_info: Dict[str, object] = {
            "wiki_nodes": wiki_nodes,
            "arxiv_nodes": arxiv_nodes,
            "similarity": similarity,
        }
        logs.append("Embedding artifacts prepared. Matrices ready for inspection.")
        return node_info, structural_info, logs

    def solve_from_inputs(
        self,
        node_info: Dict[str, object],
        structural_info: Dict[str, object],
        logs: Optional[List[str]] = None,
    ) -> PipelineResult:
        log_buffer = logs if logs is not None else []
        log_buffer.append("Solving nearest-neighbor baseline alignments...")
        nn_alignments = self._nearest_neighbor_alignments(node_info)

        log_buffer.append("Formulating QUBO problem...")
        qubo_data = formulate.formulate(
            node_info,
            structural_info,
            node_weight=QUBO_NODE_WEIGHT,
            structural_weight=QUBO_STRUCTURE_WEIGHT,
            wiki_penalty=QUBO_WIKI_PENALTY,
            arxiv_penalty=QUBO_ARXIV_PENALTY,
            similarity_threshold=None,
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
            sample, qubo_data["reverse_index"], node_info
        )
        qubo_energy = float(getattr(record, "energy"))
        log_buffer.append(f"QUBO energy: {qubo_energy:.4f} | alignments: {len(qubo_alignments)}")

        return PipelineResult(
            nn_alignments=nn_alignments,
            qubo_alignments=qubo_alignments,
            qubo_energy=qubo_energy,
            logs=log_buffer,
        )

    # ------------------------------------------------------------------
    # Dataset prep
    # ------------------------------------------------------------------
    def _build_pyg_dataset(
        self, graph: GraphModel
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

        if USE_SCIBERT_FEATURES:
            features = []
            for node in nodes:
                text = node.label
                if node.description:
                    text = f"{node.label}. {node.description}"
                embedding = get_mean_pooled_embedding(text, self.tokenizer, self.scibert)
                features.append(embedding.numpy())
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = torch.eye(num_nodes)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if wiki_data.x is None or arxiv_data.x is None:
            raise ValueError("Node feature tensors are missing; unable to run embeddings.")

        use_gae = USE_GAE_FOR_ENTITY_EMBEDDINGS
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

        if USE_SCIBERT_FEATURES and use_gae:
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
    def _nearest_neighbor_alignments(self, node_info: Dict[str, object]) -> List[AlignmentRow]:
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
    ) -> List[AlignmentRow]:
        wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]
        similarity: torch.Tensor = cast(torch.Tensor, node_info["similarity"])  # type: ignore[assignment]

        alignments: List[AlignmentRow] = []
        for var, value in sample.items():
            if not value:
                continue
            wiki_idx, arxiv_idx = reverse_index[var]
            alignments.append(
                AlignmentRow(
                    wiki=wiki_nodes[wiki_idx].label,
                    arxiv=arxiv_nodes[arxiv_idx].label,
                    similarity=float(similarity[wiki_idx, arxiv_idx]),
                )
            )
        alignments.sort(key=lambda item: item.similarity, reverse=True)
        return alignments

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())
