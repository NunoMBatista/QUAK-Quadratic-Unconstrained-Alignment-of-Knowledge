"""Alignment solvers for knowledge graph entities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import dimod
from neal import SimulatedAnnealingSampler
from rdflib import Graph, URIRef
import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

from src.config import *
from src.qubo_alignment import formulate, weights
from src.utils.graph_visualizer import visualize_ttl


ALIGNED_KG_HTML = KG_DIR / "aligned_kg.html"


@dataclass
class Alignment:
	"""single alignment between wiki and arxiv entities"""

	wiki_index: int
	arxiv_index: int
	wiki_uri: URIRef
	arxiv_uri: URIRef
	similarity: float


@dataclass
class AlignmentResult:
	"""Container for alignment solver results."""

	alignments: List[Alignment]
	energy: Optional[float]
	sampleset: Optional[dimod.SampleSet]
	aligned_graph_path: Optional[Path]
	aligned_graph_html: Optional[Path]
	alignment_report_path: Path


def _build_alignment_problem(
	similarity_threshold: Optional[float],
	node_weight: float,
	structural_weight: float,
	wiki_penalty: float,
	arxiv_penalty: float,
	max_structural_pairs: Optional[int],
):
	# prepare node and relation weights for the qubo
	node_info = weights.load_node_similarity()
	structural_info = weights.load_structural_weights(
		node_info["wiki_lookup"], node_info["arxiv_lookup"]
	)

	if max_structural_pairs is not None:
		items = list(structural_info["weights"].items())[: max_structural_pairs]
		structural_info = {**structural_info, "weights": dict(items)}

	qubo_data = formulate.formulate(
		node_info,
		structural_info,
		node_weight=node_weight,
		structural_weight=structural_weight,
		wiki_penalty=wiki_penalty,
		arxiv_penalty=arxiv_penalty,
		similarity_threshold=similarity_threshold,
	)

	if not qubo_data["Q"]:
		raise ValueError("qubo has no variables; adjust thresholds or weights")

	print(
		f"[QUBO] candidate variables: {len(qubo_data['index'])}, structural pairs: {len(structural_info['weights'])}"
	)

	return qubo_data, node_info


def _solve_with_simulated_annealing(
	qubo: Dict[Tuple[int, int], float],
	num_reads: int,
	beta_range: Optional[Tuple[float, float]],
	seed: Optional[int],
	sampler: Optional[SimulatedAnnealingSampler],
) -> dimod.SampleSet:
	# sample the qubo with dwave's annealer simulator
	bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
	solver = sampler or SimulatedAnnealingSampler()
	print(
		f"[QUBO] running simulated annealer with num_reads={num_reads}, beta_range={beta_range}, seed={seed}"
	)
	return solver.sample(bqm, num_reads=num_reads, beta_range=beta_range, seed=seed)


def _extract_alignments(
	sample: Dict[int, int],
	reverse_index: Dict[int, Tuple[int, int]],
	node_info: Dict[str, object],
) -> List[Alignment]:
	# convert binary solution into entity alignments
	similarity = cast(torch.Tensor, node_info["similarity"])
	wiki_nodes = cast(Sequence[URIRef], node_info["wiki_nodes"])
	arxiv_nodes = cast(Sequence[URIRef], node_info["arxiv_nodes"])

	results: List[Alignment] = []
	for var, value in sample.items():
		if not value:
			continue
		i, j = reverse_index[var]
		results.append(
			Alignment(
				wiki_index=i,
				arxiv_index=j,
				wiki_uri=wiki_nodes[i],
				arxiv_uri=arxiv_nodes[j],
				similarity=float(similarity[i, j]),
			)
		)

	results.sort(key=lambda item: item.similarity, reverse=True)
	return results


def _build_aligned_graph(alignments: Iterable[Alignment]) -> Graph:
	# merge both input kgs and add alignment links
	graph = Graph()
	graph.bind("ont", NS_ONT)

	wiki_graph = Graph()
	wiki_graph.parse(str(KG_WIKI_FINAL_PATH), format="turtle")
	arxiv_graph = Graph()
	arxiv_graph.parse(str(KG_ARXIV_FINAL_PATH), format="turtle")

	for triple in wiki_graph:
		graph.add(triple)
	for triple in arxiv_graph:
		graph.add(triple)

	predicate = NS_ONT["alignedWith"]
	for item in alignments:
		graph.add((item.wiki_uri, predicate, item.arxiv_uri))
		graph.add((item.arxiv_uri, predicate, item.wiki_uri))

	return graph


def _entity_label(uri: URIRef) -> str:
	# human-friendly label for reporting
	text = str(uri)
	if "#" in text:
		text = text.rsplit("#", 1)[-1]
	if "/" in text:
		text = text.rsplit("/", 1)[-1]
	return text.replace("_", " ")


def _write_alignment_report(
	alignments: List[Alignment],
	node_info: Dict[str, object],
	path: Path,
) -> Path:
	# persist csv report of aligned and unaligned entities
	path.parent.mkdir(parents=True, exist_ok=True)

	wiki_nodes = cast(Sequence[URIRef], node_info["wiki_nodes"])
	arxiv_nodes = cast(Sequence[URIRef], node_info["arxiv_nodes"])

	aligned_wiki = {item.wiki_index for item in alignments}
	aligned_arxiv = {item.arxiv_index for item in alignments}

	unaligned_entries: List[str] = []
	for idx, uri in enumerate(wiki_nodes):
		if idx not in aligned_wiki:
			unaligned_entries.append(f"wiki: {_entity_label(uri)}")
	for idx, uri in enumerate(arxiv_nodes):
		if idx not in aligned_arxiv:
			unaligned_entries.append(f"arxiv: {_entity_label(uri)}")

	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(["wiki_entity", "arxiv_entity", "not_aligned"])
		for item in alignments:
			writer.writerow([
				_entity_label(item.wiki_uri),
				_entity_label(item.arxiv_uri),
				"",
			])
		for entry in unaligned_entries:
			writer.writerow(["", "", entry])

	print(
		f"[QUBO] alignment report saved to {path} with {len(alignments)} matches and {len(unaligned_entries)} unaligned entries"
	)

	return path


def solve_alignment_with_annealer(
	*,
	similarity_threshold: Optional[float] = None,
	node_weight: float = 1.0,
	structural_weight: float = 1.0,
	wiki_penalty: float = 2.0,
	arxiv_penalty: float = 2.0,
	max_structural_pairs: Optional[int] = None,
	num_reads: int = 100,
	beta_range: Optional[Tuple[float, float]] = None,
	seed: Optional[int] = None,
	sampler: Optional[SimulatedAnnealingSampler] = None,
	visualize: bool = True,
) -> AlignmentResult:
	"""
 	solve the alignment qubo via dwave's simulated annealing.
 
	Parameters
	----------
	similarity_threshold : Optional[float]
		Minimum cosine similarity to consider a node pair.
	node_weight : float
		Weight for node similarity terms in the QUBO.
	structural_weight : float
		Weight for structural consistency terms in the QUBO.
	wiki_penalty : float
		Penalty for multiple wiki nodes aligned to the same arxiv node.
	arxiv_penalty : float
		Penalty for multiple arxiv nodes aligned to the same wiki node.
	max_structural_pairs : Optional[int]
		Optional cap on the number of structural terms (for quick sanity checks).
	num_reads : int	
		Number of reads for the simulated annealing sampler.
	beta_range : Optional[Tuple[float, float]]
		Optional beta range for the simulated annealing sampler.
	seed : Optional[int]	
		Random seed for the simulated annealing sampler.
	sampler : Optional[SimulatedAnnealingSampler]
		Custom sampler instance to use instead of the default.
	visualize : bool
		Whether to generate an HTML visualization of the aligned KG.

	Returns
	-------
	AlignmentResult
		The result of the alignment process.
 
 	"""

	qubo_data, node_info = _build_alignment_problem(
		similarity_threshold,
		node_weight,
		structural_weight,
		wiki_penalty,
		arxiv_penalty,
		max_structural_pairs,
	)

	sampleset = _solve_with_simulated_annealing(
		qubo_data["Q"], num_reads, beta_range, seed, sampler
	)

	record = sampleset.first  # type: ignore[assignment]
	energy_value = float(record.energy)  # type: ignore[attr-defined]
	sample = cast(Dict[int, int], record.sample)  # type: ignore[attr-defined]
	alignments = _extract_alignments(sample, qubo_data["reverse_index"], node_info)
	print(
		f"[QUBO] best sample energy={energy_value:.4f} produced {len(alignments)} alignments"
	)

	aligned_graph = _build_aligned_graph(alignments)
	aligned_graph.serialize(str(KG_ALIGNED_PATH), format="turtle")

	html_path: Optional[Path] = None
	if visualize:
		visualize_ttl(KG_ALIGNED_PATH, ALIGNED_KG_HTML)
		html_path = ALIGNED_KG_HTML

	report_path = _write_alignment_report(
		alignments,
		node_info,
		ALIGNED_ENTITIES_ANNEALER_CSV,
	)

	return AlignmentResult(
		alignments=alignments,
		sampleset=sampleset,
		energy=energy_value,
		aligned_graph_path=KG_ALIGNED_PATH,
		aligned_graph_html=html_path,
		alignment_report_path=report_path,
	)


def solve_alignment_with_nearest_neighbor(
	*,
	similarity_threshold: Optional[float] = None,
	visualize: bool = False,
) -> AlignmentResult:
	"""Greedy nearest-neighbor baseline using cosine similarity scores."""

	node_info = weights.load_node_similarity()
	similarity = cast(torch.Tensor, node_info["similarity"])
	wiki_nodes = cast(Sequence[URIRef], node_info["wiki_nodes"])
	arxiv_nodes = cast(Sequence[URIRef], node_info["arxiv_nodes"])

	num_wiki, num_arxiv = similarity.shape
	candidates: List[Tuple[float, int, int]] = []
	threshold = similarity_threshold
	if threshold is not None:
		threshold = float(threshold)

	# Enumerate candidate pairs and keep only those above the threshold, if provided.
	for wiki_idx in range(num_wiki):
		for arxiv_idx in range(num_arxiv):
			score = float(similarity[wiki_idx, arxiv_idx])
			if threshold is not None and score < threshold:
				continue
			candidates.append((score, wiki_idx, arxiv_idx))

	candidates.sort(key=lambda item: item[0], reverse=True)
	used_wiki: set[int] = set()
	used_arxiv: set[int] = set()
	alignments: List[Alignment] = []

	def _assign(pair_list: List[Tuple[float, int, int]], *, ignore_threshold: bool = False) -> None:
		for score, wiki_idx, arxiv_idx in pair_list:
			if wiki_idx in used_wiki or arxiv_idx in used_arxiv:
				continue
			if not ignore_threshold and threshold is not None and score < threshold:
				continue
			used_wiki.add(wiki_idx)
			used_arxiv.add(arxiv_idx)
			alignments.append(
				Alignment(
					wiki_index=wiki_idx,
					arxiv_index=arxiv_idx,
					wiki_uri=wiki_nodes[wiki_idx],
					arxiv_uri=arxiv_nodes[arxiv_idx],
					similarity=score,
				)
			)

	_assign(candidates)

	if threshold is not None:
		remaining_wiki = [idx for idx in range(num_wiki) if idx not in used_wiki]
		remaining_arxiv = [idx for idx in range(num_arxiv) if idx not in used_arxiv]
		if remaining_wiki and remaining_arxiv:
			fallback_pairs = []
			for wiki_idx in remaining_wiki:
				for arxiv_idx in range(num_arxiv):
					if arxiv_idx in used_arxiv:
						continue
					fallback_pairs.append((float(similarity[wiki_idx, arxiv_idx]), wiki_idx, arxiv_idx))
			fallback_pairs.sort(key=lambda item: item[0], reverse=True)
			_assign(fallback_pairs, ignore_threshold=True)

	alignments.sort(key=lambda item: item.similarity, reverse=True)

	print(
		f"[NN] produced {len(alignments)} alignments from {num_wiki}×{num_arxiv} similarity scores"
	)

	report_path = _write_alignment_report(
		alignments,
		node_info,
		ALIGNED_ENTITIES_NN_CSV,
	)

	aligned_graph_path: Optional[Path] = None
	aligned_graph_html: Optional[Path] = None
	if visualize and alignments:
		aligned_graph = _build_aligned_graph(alignments)
		aligned_graph_path = KG_ALIGNED_PATH
		aligned_graph.serialize(str(aligned_graph_path), format="turtle")
		visualize_ttl(aligned_graph_path, ALIGNED_KG_HTML)
		aligned_graph_html = ALIGNED_KG_HTML

	return AlignmentResult(
		alignments=alignments,
		sampleset=None,
		energy=None,
		aligned_graph_path=aligned_graph_path,
		aligned_graph_html=aligned_graph_html,
		alignment_report_path=report_path,
	)


if __name__ == "__main__":
	result = solve_alignment_with_annealer(
		similarity_threshold=0.0,
		max_structural_pairs=2000,
		num_reads=50,
		visualize=False,
	)

	print(f"anneal energy: {result.energy:.4f}")
	print(f"alignments found: {len(result.alignments)}")
	for item in result.alignments[:10]:
		wiki = item.wiki_uri
		arxiv = item.arxiv_uri
		score = item.similarity
		print(f"  {wiki} <-> {arxiv} (sim={score:.4f})")
	print(f"aligned kg saved to: {result.aligned_graph_path}")
	if result.aligned_graph_html:
		print(f"visualization: {result.aligned_graph_html}")
	print(f"alignment report: {result.alignment_report_path}")
