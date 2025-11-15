"""LLM-driven KG extraction pipeline using Groq chat completions."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from groq import Groq, BadRequestError
from rdflib import Graph

from src.config import (
	ARXIV_ENTITIES_DIR,
	ARXIV_ENTITIES_FILE,
	DOTENV_PATH,
	KG_ARXIV_UNPRUNED_PATH,
	KG_WIKI_UNPRUNED_PATH,
	LLM_CHUNK_CHAR_LIMIT,
	LLM_EXTRACTION_JSON_SCHEMA,
	LLM_EXTRACTION_SYSTEM_PROMPT,
	LLM_EXTRACTION_USER_PROMPT,
	LLM_MAX_ENTITIES_PER_CHUNK,
	LLM_MAX_OUTPUT_TOKENS,
	LLM_MAX_TRIPLES_PER_CHUNK,
	LLM_MODEL_NAME,
	LLM_RECON_JSON_SCHEMA,
	LLM_RECON_MAX_NAMES,
	LLM_RECON_SYSTEM_PROMPT,
	LLM_RECON_USER_PROMPT,
	LLM_TEMPERATURE,
	NS_RAW,
	REL_GENERIC,
	WIKI_ENTITIES_DIR,
	WIKI_ENTITIES_FILE,
)


_SCHEMA_RETRY_LIMIT = 2


@dataclass(frozen=True)
class LLMEntity:
	name: str
	type: str = "Other"
	description: str = ""


@dataclass(frozen=True)
class LLMTriple:
	subject: str
	relation: str
	object: str
	evidence: Optional[str] = None
	confidence: Optional[float] = None


@dataclass
class CorpusExtraction:
	raw_entities: List[LLMEntity]
	triples: List[LLMTriple]
	chunk_records: List[Dict[str, Any]]


class LLMBasedPipeline:
	"""Minimal Groq-powered KG builder that mirrors the NLPPipeline surface."""

	def __init__(
		self,
		*,
		model_name: str = LLM_MODEL_NAME,
		temperature: float = LLM_TEMPERATURE,
		max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
		chunk_char_limit: int = LLM_CHUNK_CHAR_LIMIT,
		max_entities_per_chunk: int = LLM_MAX_ENTITIES_PER_CHUNK,
		max_triples_per_chunk: int = LLM_MAX_TRIPLES_PER_CHUNK,
		client: Optional[Groq] = None,
	) -> None:
		load_dotenv(dotenv_path=DOTENV_PATH, override=False)
		api_key = os.getenv("GROQ_API_KEY")
		if not api_key:
			raise RuntimeError(
				"Missing GROQ_API_KEY. Please add it to .env at the project root."
			)

		self.client = client or Groq(api_key=api_key)
		self.model_name = model_name
		self.temperature = temperature
		self.max_output_tokens = max_output_tokens
		self.chunk_char_limit = chunk_char_limit
		self.max_entities_per_chunk = max_entities_per_chunk
		self.max_triples_per_chunk = max_triples_per_chunk
		self.schema_retry_limit = _SCHEMA_RETRY_LIMIT
		self._extraction_response_format = {
			"type": "json_schema",
			"json_schema": {
				"name": "kg_extraction",
				"schema": LLM_EXTRACTION_JSON_SCHEMA,
				"strict": True,
			},
		}
		self._recon_response_format = {
			"type": "json_schema",
			"json_schema": {
				"name": "kg_entity_reconciliation",
				"schema": LLM_RECON_JSON_SCHEMA,
				"strict": True,
			},
		}

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def build_unpruned_kgs(self, wiki_data: Sequence[str], arxiv_data: Sequence[str]) -> None:
		"""Entry-point that mirrors build_kg.build_unpruned_kgs but uses the LLM."""

		#print("\n--- STEP 1: RUN THE LLM PIPELINE ---")
		wiki_result = self._process_corpus("wiki", wiki_data)
		arxiv_result = self._process_corpus("arxiv", arxiv_data)
		wiki_entities, wiki_triples = self._canonicalize_corpus("wiki", wiki_result)
		arxiv_entities, arxiv_triples = self._canonicalize_corpus("arxiv", arxiv_result)

		print("\n--- STEP 2: BUILD THE UNPRUNED KGs (LLM) ---")
		self._persist_corpus(
			corpus_label="wiki",
			entities=wiki_entities,
			triples=wiki_triples,
			ttl_path=KG_WIKI_UNPRUNED_PATH,
			entity_dir=WIKI_ENTITIES_DIR,
			entity_file=WIKI_ENTITIES_FILE,
		)
		self._persist_corpus(
			corpus_label="arXiv",
			entities=arxiv_entities,
			triples=arxiv_triples,
			ttl_path=KG_ARXIV_UNPRUNED_PATH,
			entity_dir=ARXIV_ENTITIES_DIR,
			entity_file=ARXIV_ENTITIES_FILE,
		)

	# ------------------------------------------------------------------
	# Core processing
	# ------------------------------------------------------------------
	def _process_corpus(self, corpus_name: str, documents: Sequence[str]) -> CorpusExtraction:
		result = CorpusExtraction(raw_entities=[], triples=[], chunk_records=[])
		total_docs = len(documents)
		for doc_index, raw_text in enumerate(documents):
			normalized = (raw_text or "").strip()
			if not normalized:
				continue
			chunks = self._chunk_text(normalized)
			print(
				f"[LLM Pipeline] {corpus_name} document {doc_index + 1}/{total_docs} -> {len(chunks)} chunk(s)"
			)
			for chunk_idx, chunk in enumerate(chunks):
				print(
					f"   - Chunk {chunk_idx + 1}/{len(chunks)} (chars: {len(chunk)})"
				)
				payload = self._build_prompt_chunk(
					text=chunk,
					source_id=f"{corpus_name}_{doc_index}",
					chunk_index=chunk_idx,
					total_chunks=len(chunks),
				)
				response = self._invoke_llm(
					user_prompt=payload,
					system_prompt=LLM_EXTRACTION_SYSTEM_PROMPT,
					response_format=self._extraction_response_format,
				)
				block = self._parse_extraction_response(response)
				if not block:
					continue
				sanitized_triples = self._sanitize_triples(block["triples"])
				meta = {
					"source_id": f"{corpus_name}_{doc_index}",
					"chunk_index": chunk_idx,
					"total_chunks": len(chunks),
				}
				result.raw_entities.extend(block["entities"])
				result.triples.extend(sanitized_triples)
				result.chunk_records.append(
					{
						"meta": meta,
						"entities": block["entities"],
						"triples": sanitized_triples,
					}
				)

		unique_entity_count = len(
			{
				entity.name.strip().lower()
				for entity in result.raw_entities
				if entity.name.strip()
			}
		)
		print(f"-> {corpus_name} raw entities: {unique_entity_count} | triples: {len(result.triples)}")
		return result

	def _canonicalize_corpus(
		self,
		corpus_label: str,
		extraction: CorpusExtraction,
	) -> Tuple[List[LLMEntity], List[LLMTriple]]:
		all_names = sorted(
			{
				entity.name.strip()
				for entity in extraction.raw_entities
				if entity.name.strip()
			}
		)
		alias_lookup = self._reconcile_entities(all_names)

		canonical_entities: Dict[str, LLMEntity] = {}
		for entity in extraction.raw_entities:
			canonical_name = self._canonical_name(entity.name, alias_lookup)
			if not canonical_name:
				continue
			key = canonical_name.lower()
			existing = canonical_entities.get(key)
			type_value = (existing.type if existing and existing.type else "") or entity.type or "Other"
			description_value = existing.description if existing and existing.description else entity.description
			canonical_entities[key] = LLMEntity(
				name=canonical_name,
				type=type_value,
				description=description_value or "",
			)

		canonical_triples: List[LLMTriple] = []
		for triple in extraction.triples:
			subject = self._canonical_name(triple.subject, alias_lookup)
			object_name = self._canonical_name(triple.object, alias_lookup)
			if not subject or not object_name:
				continue
			canonical_triples.append(
				LLMTriple(
					subject=subject,
					relation=_clean_relation_label(triple.relation),
					object=object_name,
					evidence=triple.evidence,
					confidence=triple.confidence,
				)
			)

		print(
			f"-> {corpus_label} canonical entities: {len(canonical_entities)} | triples: {len(canonical_triples)}"
		)
		return list(canonical_entities.values()), canonical_triples

	def _reconcile_entities(self, names: Sequence[str]) -> Dict[str, str]:
		if not names:
			return {}

		alias_lookup: Dict[str, str] = {}
		for batch in _chunk_list(list(names), LLM_RECON_MAX_NAMES):
			if not batch:
				continue
			payload = LLM_RECON_USER_PROMPT.replace(
				"{entity_names}", json.dumps(batch, indent=2)
			)
			response = self._invoke_llm(
				user_prompt=payload,
				system_prompt=LLM_RECON_SYSTEM_PROMPT,
				response_format=self._recon_response_format,
			)
			mappings = self._parse_reconciliation_response(response)
			for mapping in mappings:
				canonical = mapping.get("canonical", "").strip()
				if not canonical:
					continue
				aliases = mapping.get("aliases", []) or []
				aliases.append(canonical)
				for alias in aliases:
					alias_clean = alias.strip()
					if not alias_clean:
						continue
					alias_lookup[alias_clean.lower()] = canonical
		return alias_lookup

	def _parse_reconciliation_response(self, raw_content: str) -> List[Dict[str, Any]]:
		if not raw_content:
			return []
		json_payload = self._extract_json(raw_content)
		if not json_payload:
			return []
		try:
			data = json.loads(json_payload)
		except json.JSONDecodeError:
			return []
		payload = data.get("canonical_entities")
		if isinstance(payload, list):
			return payload
		return []
	def _canonical_name(self, name: str, alias_lookup: Dict[str, str]) -> str:
		clean = name.strip()
		if not clean:
			return ""
		return alias_lookup.get(clean.lower(), clean)

	def _sanitize_triples(self, triples: Sequence[LLMTriple]) -> List[LLMTriple]:
		cleaned: List[LLMTriple] = []
		for triple in triples:
			subject = triple.subject.strip()
			obj = triple.object.strip()
			if not subject or not obj:
				continue
			relation = _clean_relation_label(triple.relation or "related_to")
			cleaned.append(
				LLMTriple(
					subject=subject,
					relation=relation,
					object=obj,
					evidence=triple.evidence,
					confidence=triple.confidence,
				)
			)
		return cleaned

	def _chunk_text(self, text: str) -> List[str]:
		if len(text) <= self.chunk_char_limit:
			return [text]

		chunks: List[str] = []
		start = 0
		while start < len(text):
			end = min(start + self.chunk_char_limit, len(text))
			if end < len(text):
				newline = text.rfind("\n", start, end)
				if newline > start + 100:  # keep sizable chunk
					end = newline
			chunks.append(text[start:end].strip())
			start = end
		return [chunk for chunk in chunks if chunk]

	def _build_prompt_chunk(
		self,
		*,
		text: str,
		source_id: str,
		chunk_index: int,
		total_chunks: int,
	) -> str:
		decorated_text = (
			f"Source ID: {source_id}\n"
			f"Chunk: {chunk_index + 1}/{total_chunks}\n\n"
			f"{text.strip()}"
		)
		return LLM_EXTRACTION_USER_PROMPT.replace("{text}", decorated_text)

	def _invoke_llm(
		self,
		*,
		user_prompt: str,
		system_prompt: str,
		response_format: Optional[Dict[str, Any]] = None,
		force_json: bool = True,
	) -> str:
		params: Dict[str, Any] = {
			"model": self.model_name,
			"temperature": self.temperature,
			"max_tokens": self.max_output_tokens,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		}
		if response_format is not None:
			params["response_format"] = response_format
		elif force_json:
			params["response_format"] = {"type": "json_object"}

		messages = list(params["messages"])
		params["messages"] = messages
		for attempt in range(self.schema_retry_limit + 1):
			try:
				response = self.client.chat.completions.create(**params)
				content = response.choices[0].message.content or ""
				return content.strip()
			except BadRequestError as error:
				if not self._should_retry_schema(error) or attempt == self.schema_retry_limit:
					self._log_groq_schema_error(error)
					raise RuntimeError(
						"Groq rejected the response as non-compliant with the configured JSON schema."
					) from error
				extra_instruction = self._build_schema_retry_instruction(error)
				messages.append({"role": "system", "content": extra_instruction})

		raise RuntimeError("LLM invocation exhausted without producing a valid response.")

	def _parse_extraction_response(self, raw_content: str) -> Optional[Dict[str, List]]:
		if not raw_content:
			return None
		json_payload = self._extract_json(raw_content)
		if not json_payload:
			return None
		try:
			data = json.loads(json_payload)
		except json.JSONDecodeError:
			return None

		entities = [
			LLMEntity(
				name=item.get("name", "").strip(),
				type=item.get("type", "Other").strip() or "Other",
				description=item.get("description", "").strip(),
			)
			for item in (data.get("entities") or [])[: self.max_entities_per_chunk]
		]

		triples = [
			LLMTriple(
				subject=item.get("subject", "").strip(),
				relation=item.get("relation", "related_to").strip(),
				object=item.get("object", "").strip(),
				evidence=item.get("evidence"),
				confidence=_safe_float(item.get("confidence")),
			)
			for item in (data.get("triples") or [])[: self.max_triples_per_chunk]
		]

		return {"entities": entities, "triples": triples}

	@staticmethod
	def _should_retry_schema(error: BadRequestError) -> bool:
		details = _extract_groq_error_details(error)
		return (details.get("code") == "json_validate_failed")

	@staticmethod
	def _build_schema_retry_instruction(error: BadRequestError) -> str:
		details = _extract_groq_error_details(error)
		message = details.get("message", "the previous response violated the JSON schema")
		missing_hint = ""
		if "missing properties" in message:
			missing_hint = " It must always include `entities` and `triples`, using [] when empty."
		return (
			"Reminder: The previous JSON response was rejected because "
			+ message
			+ ". Provide valid JSON that follows the configured schema strictly."
			+ missing_hint
		)

	@staticmethod
	def _log_groq_schema_error(error: BadRequestError) -> None:
		message = getattr(error, "message", str(error))
		status_code = getattr(error, "status_code", None)
		body = getattr(error, "body", None)
		response_obj = getattr(error, "response", None)
		response_text = None
		if response_obj is not None:
			response_text = getattr(response_obj, "text", None)
			if response_text is None:
				try:
					response_text = response_obj.json()
				except Exception:  # pragma: no cover
					response_text = str(response_obj)
		print(
			"[LLM Pipeline] Groq schema rejection.",
			"Status:", status_code,
			"Message:", message,
			"\nBody:", body,
			"\nResponse:", response_text,
			file=sys.stderr,
		)

	@staticmethod
	def _extract_json(text: str) -> Optional[str]:
		text = text.strip()
		if text.startswith("{") and text.endswith("}"):
			return text
		code_fence_match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
		if code_fence_match:
			return code_fence_match.group(1).strip()
		brace_match = re.search(r"\{.*\}", text, re.DOTALL)
		if brace_match:
			return brace_match.group(0)
		return None

	# ------------------------------------------------------------------
	# Persistence helpers
	# ------------------------------------------------------------------
	def _persist_corpus(
		self,
		*,
		corpus_label: str,
		entities: Sequence[LLMEntity],
		triples: Sequence[LLMTriple],
		ttl_path: Path,
		entity_dir: Path,
		entity_file: Path,
	) -> None:
		_persist_entities(entities, entity_dir, entity_file, corpus_label)
		graph = self._triples_to_graph(triples)
		ttl_path.parent.mkdir(parents=True, exist_ok=True)
		graph.serialize(destination=str(ttl_path), format="turtle")
		print(f"-> Saved {len(graph)} {corpus_label} triples to {ttl_path}")

	def _triples_to_graph(self, triples: Sequence[LLMTriple]) -> Graph:
		graph = Graph()
		graph.bind("raw", NS_RAW)
		for triple in triples:
			subj_label = _clean_for_uri(triple.subject)
			obj_label = _clean_for_uri(triple.object)
			if not subj_label or not obj_label:
				continue
			predicate = _relation_uri(triple.relation) if triple.relation else REL_GENERIC
			graph.add((NS_RAW[subj_label], predicate, NS_RAW[obj_label]))
		return graph


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _clean_for_uri(text: str) -> str:
	text = text.replace(" ", "_")
	text = re.sub(r"[^a-zA-Z0-9_'\-]", "", text)
	return text


def _clean_relation_label(text: str) -> str:
	clean = re.sub(r"[^a-z0-9_]+", "_", text.lower())
	clean = re.sub(r"_+", "_", clean).strip("_")
	return clean or "related_to"


def _relation_uri(value: str):
	label = _clean_for_uri(value) or "related_to"
	return NS_RAW[label]


def _chunk_list(values: List[str], chunk_size: int) -> List[List[str]]:
	if chunk_size <= 0:
		return [values]
	return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _persist_entities(
	entities: Iterable[LLMEntity], output_dir: Path, output_file: Path, label: str
) -> None:
	canonical: Dict[str, LLMEntity] = {}
	for entity in entities:
		name = entity.name.strip()
		if not name:
			continue
		key = name.lower()
		if key not in canonical:
			canonical[key] = entity

	sorted_names = sorted(canonical.values(), key=lambda ent: ent.name.lower())
	output_dir.mkdir(parents=True, exist_ok=True)
	content = "\n".join(ent.name for ent in sorted_names)
	output_file.write_text(content + ("\n" if content else ""), encoding="utf-8")
	print(f"-> Saved {len(sorted_names)} {label} entities to {output_file}")


def _safe_float(value) -> Optional[float]:
	try:
		if value is None:
			return None
		return float(value)
	except (TypeError, ValueError):
		return None


def _extract_groq_error_details(error: BadRequestError) -> Dict[str, Any]:
	body = getattr(error, "body", None)
	if isinstance(body, dict):
		error_payload = body.get("error")
		if isinstance(error_payload, dict):
			return error_payload
		return body
	if isinstance(body, str):
		try:
			parsed = json.loads(body)
		except json.JSONDecodeError:
			return {"message": body}
		if isinstance(parsed, dict):
			error_payload = parsed.get("error")
			if isinstance(error_payload, dict):
				return error_payload
			return parsed
	return {}


if __name__ == "__main__":
	print("This module is intended to be imported by the KG builder scripts.")
