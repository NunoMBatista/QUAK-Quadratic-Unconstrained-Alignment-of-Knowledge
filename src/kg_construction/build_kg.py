import re
from tarfile import NUL
from src.config import *
from src.kg_construction.fetch_data import fetch_wiki_data, fetch_arxiv_data
from src.kg_construction.nlp_pipeline import NLPPipeline
from rdflib import Graph, Namespace
from src.utils.graph_visualizer import visualize_ttl
from pathlib import Path


def clean_for_uri(text):
    """
    Cleans a raw string from SciBERT to be a valid URI fragment.
    1. Replaces spaces with underscores.
    2. Removes any characters that are not letters, numbers,
       underscores, or hyphens.
    """
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Remove any invalid characters
    text = re.sub(r'[^a-zA-Z0-9_-]', '', text)
    return text

def prune_kgs():
    pass

def build_unpruned_kgs():
    """
    this will build the unpruned KGs for both Wikipedia and
    arXiv data...
    """
    
    print("\n--- STEP 1: FETCH THE DATA ---")
    # Using your static config lists now, which is great
    wiki_data = fetch_wiki_data()
    arxiv_data = fetch_arxiv_data()
    
    print("\n--- STEP 2: RUN THE NLP PIPELINE ---")
    nlp_pipeline = NLPPipeline()
    wiki_triples = []
    for article in wiki_data:
        triples = nlp_pipeline.extract_triples(article)
        wiki_triples.extend(triples)
    arxiv_triples = []
    for abstract in arxiv_data:
        triples = nlp_pipeline.extract_triples(abstract)
        arxiv_triples.extend(triples)
        
    print("\n--- STEP 3: BUILD THE UNPRUNED KGs ---")
    print(f"-> Wiki triples extracted: {len(wiki_triples)}")
    print(f"-> arXiv triples extracted: {len(arxiv_triples)}")

    # Build the unpruned KGs
    wiki_kg = Graph()
    arxiv_kg = Graph()
    
    # --- HERE IS THE FIX ---
    # Added the [:5] slice back in for quick testing
    for subj, pred, obj in wiki_triples[:MAX_UNPRUNED_TRIPLES]:
        # Clean the strings before adding them
        subj_clean = clean_for_uri(subj)
        obj_clean = clean_for_uri(obj)
        wiki_kg.add((NS_RAW[subj_clean], REL_GENERIC, NS_RAW[obj_clean]))
        
    for subj, pred, obj in arxiv_triples[:MAX_UNPRUNED_TRIPLES]:
        # Clean the strings before adding them
        subj_clean = clean_for_uri(subj)
        obj_clean = clean_for_uri(obj)
        arxiv_kg.add((NS_RAW[subj_clean], REL_GENERIC, NS_RAW[obj_clean]))
    # -----------------------
    
    # Save the unpruned KGs to files
    print(f"\nSaving {len(wiki_kg)} wiki triples to {KG_WIKI_UNPRUNED_PATH}")
    wiki_kg.serialize(destination=str(KG_WIKI_UNPRUNED_PATH), format='turtle')
    
    print(f"Saving {len(arxiv_kg)} arXiv triples to {KG_ARXIV_UNPRUNED_PATH}")
    arxiv_kg.serialize(destination=str(KG_ARXIV_UNPRUNED_PATH), format='turtle')

if __name__ == "__main__":
    KG_DIR.mkdir(exist_ok=True, parents=True)
    build_unpruned_kgs()

    print("\n--- VISUALIZING (no checks) ---")
    visualize_ttl(KG_WIKI_UNPRUNED_PATH,  KG_DIR / 'unpruned_wiki_kg.html')
    visualize_ttl(KG_ARXIV_UNPRUNED_PATH, KG_DIR / 'unpruned_arxiv_kg.html')