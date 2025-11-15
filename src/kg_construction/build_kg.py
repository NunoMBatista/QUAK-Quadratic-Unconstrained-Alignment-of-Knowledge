import re
from typing import List, Set, Tuple, cast
from src.config import *
from src.kg_construction.fetch_data import fetch_wiki_data, fetch_arxiv_data
from src.kg_construction.nlp_pipeline import NLPPipeline, RelationTriple
from src.kg_construction.llm_based_pipeline import LLMBasedPipeline
from rdflib import Graph, RDF
from src.utils.graph_visualizer import visualize_ttl
from pathlib import Path


def clean_for_uri(text):
    """
    cleans a raw string from SciBERT to be a valid URI fragment.
    1. replaces spaces with underscores.
    2. removes any characters that are not letters, numbers, underscores, or hyphens.
    """
    # replace spaces with underscores
    text = text.replace(' ', '_')
    # remove any invalid characters
    # --- FIX 1: Allow apostrophes in the URI ---
    text = re.sub(r"[^a-zA-Z0-9_'-]", '', text)
    return text


def _log_doc_progress(corpus_label: str, index: int, total: int, text: str) -> None:
    preview = re.sub(r"\s+", " ", (text or "").strip())
    if len(preview) > 80:
        preview = preview[:80] + "..."
    if not preview:
        preview = "<empty document>"
    print(f"   [{corpus_label} {index + 1}/{total}] {preview}")

def _persist_entities(entities: Set[str], output_dir: Path, output_file: Path, label: str) -> None:
    canonical = {}
    for entity in entities:
        stripped = entity.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key not in canonical:
            canonical[key] = stripped

    sorted_entities = sorted(canonical.values(), key=lambda value: value.lower())

    output_dir.mkdir(parents=True, exist_ok=True)
    if sorted_entities:
        output_file.write_text("\n".join(sorted_entities) + "\n", encoding="utf-8")
    else:
        output_file.write_text("", encoding="utf-8")
    print(f"-> Saved {len(sorted_entities)} {label} entities to {output_file}")

def prune_kgs():
    """
    loads the "messy" unpruned KGs and creates clean, pruned
    KGs while preserving the automatically extracted relations.
    """
    
    print("\n--- STEP 4: PRUNING KGs ---")

    # process Wiki KG
    print(f"\n-> pruning Wiki KG...")
    _prune_and_map_kg(
        KG_WIKI_UNPRUNED_PATH,
        KG_WIKI_FINAL_PATH,
        ENTITY_MAP_WIKI
    )
    
    # process arXiv KG
    print(f"\n-> pruning arXiv KG...")
    _prune_and_map_kg(
        KG_ARXIV_UNPRUNED_PATH,
        KG_ARXIV_FINAL_PATH,
        ENTITY_MAP_ARXIV
    )

def _prune_and_map_kg(raw_path, clean_path, entity_map):
    """helper function to prune a single KG."""
    
    # load the "messy" graph
    g_raw = Graph()
    g_raw.parse(str(raw_path), format="turtle")
    
    # create the "clean" graph
    g_clean = Graph()
    
    # this binds the ontology namespace to the clean graph
    g_clean.bind("ont", NS_ONT) 
    g_clean.bind("raw", NS_RAW)
    
    # create the filter and lookup maps
    allowed_entities_lower = { # this will have all of the allowed entity names in lowercase
            name.lower() for name in entity_map.keys()
        }
    lookup_map = { # this maps the lower case names to their canonical names and types
        name.lower(): (name, entity_map[name]) for name in entity_map
        }
    
    print(f"    -> loaded {len(g_raw)} raw triples, pruning with {len(allowed_entities_lower)} entities.")
    
    # PASS 1: add all NODES (entities) that are in the allow-list
    nodes_added = set()
    for s_raw_uri, p_raw_uri, o_raw_uri in g_raw:
        s_raw_label = str(s_raw_uri).split('/')[-1].replace('_', ' ') # get the raw text
        o_raw_label = str(o_raw_uri).split('/')[-1].replace('_', ' ') # get the raw text
        
        s_low = s_raw_label.lower()
        o_low = o_raw_label.lower()

        # check if subject is allowed and is not already in the graph
        if (s_low in allowed_entities_lower) and (s_low not in nodes_added):
            s_canon, s_type = lookup_map[s_low] # get canonical name and type
            g_clean.add((NS_ONT[clean_for_uri(s_canon)], RDF.type, NS_ONT[s_type])) # add rdf:type triple
            nodes_added.add(s_low) # mark as added
        
        # check if object is allowed and is not already in the graph
        if (o_low in allowed_entities_lower) and (o_low not in nodes_added):
            o_canon, o_type = lookup_map[o_low]
            g_clean.add((NS_ONT[clean_for_uri(o_canon)], RDF.type, NS_ONT[o_type]))
            nodes_added.add(o_low)

    print(f"    -> Found and added {len(nodes_added)} matching entities.")

    # PASS 2: add all RELATIONS between the nodes we just added
    for s_raw_uri, p_raw_uri, o_raw_uri in g_raw:
        s_raw_label = str(s_raw_uri).split('/')[-1].replace('_', ' ')
        o_raw_label = str(o_raw_uri).split('/')[-1].replace('_', ' ')
        
        s_low = s_raw_label.lower()
        o_low = o_raw_label.lower()
        
        # pruning Filter (now just for relations):
        if (s_low in allowed_entities_lower) and (o_low in allowed_entities_lower):

            # get the "canonical" (properly cased) names and types
            s_canon, s_type = lookup_map[s_low]
            o_canon, o_type = lookup_map[o_low]
            
            # 'rdf:type' triples are already added. skip to relations.
            
            predicate = p_raw_uri
            g_clean.add((NS_ONT[clean_for_uri(s_canon)], predicate, NS_ONT[clean_for_uri(o_canon)]))
                
    # save the final, clean grapherarchy langu
    print(f"    -> Saving {len(g_clean)} clean triples to {clean_path}")
    g_clean.serialize(destination=str(clean_path), format="turtle")

def _relation_uri(value: str):
    """helper to create a relation URI from a raw string."""
    relation_label = clean_for_uri(value)
    if not relation_label:
        relation_label = "related_to"
    return NS_RAW[relation_label]


def build_unpruned_kgs(
    wiki_data: List[str],
    arxiv_data: List[str],
    *,
    force_rebuild: bool = False,
):
    """
    this will build the unpruned KGs for both Wikipedia and
    arXiv data.
    
    it will save them to disk as turtle files (.ttl), no need to return them.
    
    Parameters
    ----------
    wiki_data : List[str]
        list of raw Wikipedia article texts.
    arxiv_data : List[str]
        list of raw arXiv abstract texts.
    
    Parameters
    ----------
    force_rebuild : bool, optional
        Force regeneration even if cached TTLs already exist. Defaults to False.

    Returns
    -------
    None
    """
    
    if not force_rebuild:
        wiki_cached = KG_WIKI_UNPRUNED_PATH.exists()
        arxiv_cached = KG_ARXIV_UNPRUNED_PATH.exists()
        if wiki_cached and arxiv_cached:
            print("\n--- STEP 1: SKIPPING BUILD; UNPRUNED TTLs ARE ALREADY PRESENT ---")
            return
    
    mode = (KG_CONSTRUCTION_MODE or "nlp").strip().lower()
    if mode == "llm":
        print("\n--- STEP 1: RUN THE LLM PIPELINE ---")
        LLMBasedPipeline().build_unpruned_kgs(wiki_data, arxiv_data)
        return
    if mode != "nlp":
        raise ValueError(
            f"Unknown KG_CONSTRUCTION_MODE '{KG_CONSTRUCTION_MODE}'. Use 'nlp' or 'llm'."
        )

    print("\n--- STEP 1: RUN THE NLP PIPELINE ---")

    nlp_pipeline = NLPPipeline()  # initialize the NLP pipeline

    # Extract everything in both corpora (NER and RE)
    wiki_triples: List[RelationTriple] = []
    wiki_entities: Set[str] = set()
    for idx, article in enumerate(wiki_data):
        _log_doc_progress("wiki", idx, len(wiki_data), article)
        triples_result = nlp_pipeline.extract_triples(article, return_entities=True)
        triples, entities = cast(Tuple[List[RelationTriple], List[str]], triples_result)
        wiki_triples.extend(triples)
        wiki_entities.update(entities)

    arxiv_triples: List[RelationTriple] = []
    arxiv_entities: Set[str] = set()
    for idx, abstract in enumerate(arxiv_data):
        _log_doc_progress("arXiv", idx, len(arxiv_data), abstract)
        triples_result = nlp_pipeline.extract_triples(abstract, return_entities=True)
        triples, entities = cast(Tuple[List[RelationTriple], List[str]], triples_result)
        arxiv_triples.extend(triples)
        arxiv_entities.update(entities)

    _persist_entities(wiki_entities, WIKI_ENTITIES_DIR, WIKI_ENTITIES_FILE, "wiki")
    _persist_entities(arxiv_entities, ARXIV_ENTITIES_DIR, ARXIV_ENTITIES_FILE, "arXiv")

    print("\n--- STEP 2: BUILD THE UNPRUNED KGs ---")
    wiki_relation_count = len({triple.relation for triple in wiki_triples})
    arxiv_relation_count = len({triple.relation for triple in arxiv_triples})
    print(f"-> Wiki triples extracted: {len(wiki_triples)}")
    print(f"-> arXiv triples extracted: {len(arxiv_triples)}")
    print(f"-> Unique wiki relations: {wiki_relation_count}")
    print(f"-> Unique arXiv relations: {arxiv_relation_count}")

    # Build the unpruned KGs
    wiki_kg = Graph()
    arxiv_kg = Graph()

    # [:MAX_UNPRUNED_TRIPLES] for quick testing

    for triple in wiki_triples[:MAX_UNPRUNED_TRIPLES]:
        subj_clean = clean_for_uri(triple.subject)
        obj_clean = clean_for_uri(triple.object)
        wiki_kg.add((NS_RAW[subj_clean], _relation_uri(triple.relation), NS_RAW[obj_clean]))

    for triple in arxiv_triples[:MAX_UNPRUNED_TRIPLES]:
        subj_clean = clean_for_uri(triple.subject)
        obj_clean = clean_for_uri(triple.object)
        arxiv_kg.add((NS_RAW[subj_clean], _relation_uri(triple.relation), NS_RAW[obj_clean]))

    # -----------------------

    # save the unpruned KGs to files
    print(f"\n-> Saving {len(wiki_kg)} wiki triples to {KG_WIKI_UNPRUNED_PATH}")
    wiki_kg.serialize(destination=str(KG_WIKI_UNPRUNED_PATH), format='turtle')

    print(f"-> Saving {len(arxiv_kg)} arXiv triples to {KG_ARXIV_UNPRUNED_PATH}")
    arxiv_kg.serialize(destination=str(KG_ARXIV_UNPRUNED_PATH), format='turtle')


if __name__ == "__main__":
    KG_DIR.mkdir(exist_ok=True, parents=True)

    wiki_data, _wiki_titles = fetch_wiki_data()
    arxiv_data, _arxiv_ids = fetch_arxiv_data()
    
    build_unpruned_kgs(wiki_data, arxiv_data) # get the unpruned KGs into the disk
    prune_kgs() # prune them

    print("\n--- VISUALIZING RAW ---")
    visualize_ttl(KG_WIKI_UNPRUNED_PATH,  KG_DIR / 'unpruned_wiki_kg.html')
    visualize_ttl(KG_ARXIV_UNPRUNED_PATH, KG_DIR / 'unpruned_arxiv_kg.html')
    
    print("\n--- VISUALIZING PRUNED ---")
    visualize_ttl(KG_WIKI_FINAL_PATH,  KG_DIR / 'pruned_wiki_kg.html')
    visualize_ttl(KG_ARXIV_FINAL_PATH, KG_DIR / 'pruned_arxiv_kg.html')