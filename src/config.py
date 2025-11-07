from pathlib import Path
from rdflib import Namespace

# the project root is two directories above the config file
PROJECT_ROOT = Path(__file__).parent.parent


# protege files
PROTEGE_DIR = PROJECT_ROOT / 'protege'
# specific ontology file
ONTOLOGY_PATH = PROTEGE_DIR / 'ontology.ttl'


# --- pipeline outputs ---
OUTPUT_DIR = PROJECT_ROOT / 'output'
ONTOLOGY_DIR = OUTPUT_DIR / 'ontologies'

# raw data outputs
ARTICLES_DIR = OUTPUT_DIR / 'articles'
RAW_WIKI_DIR = ARTICLES_DIR / 'wiki'
RAW_ARXIV_DIR = ARTICLES_DIR / 'arxiv'

# unpruned KG outputs
KG_DIR = OUTPUT_DIR / 'KGs'
KG_WIKI_UNPRUNED_PATH = KG_DIR / 'kg_wiki_unpruned.ttl'
KG_ARXIV_UNPRUNED_PATH = KG_DIR / 'kg_arxiv_unpruned.ttl'
KG_WIKI_FINAL = KG_DIR / 'kg_wiki_final.ttl'
KG_ARXIV_FINAL = KG_DIR / 'kg_arxiv_final.ttl'


# --- visualization settings ---

# physics configuration for graph layout, it's used in /src/utils/graph_visualizer.py
GRAPH_PHYSICS = """
    {
        "enabled": true,
        "barnesHut": {
        "gravitationalConstant": -8000,
        "centralGravity": 0.5,
        "springLength": 100,
        "springConstant": 0.04,
        "damping": 0.09,
        "avoidOverlap": 0
        },
        "solver": "barnesHut",
        "minVelocity": 0.75
    }
"""

# --- NLP configuration ---

NLP_MODEL = "en_core_sci_scibert"
NLP_PIPELINE = "src/kg_construction/nlp_pipeline.py"


# --- entities and domain configuration ---

# 1. The query for Wikipedia and arXiv

# this defines if the data fetchers will search for the best matches or follow a pre-defined list of articles
QUERY_MODE = 'search' # can be either 'search' or 'fetch'

# used for the search mode
DOMAIN_QUERY = "Quantum Computing"
# how many articles to fetch in the search mode
NUM_ARTICLES = 2 

# these are both used for the fetch query mode
WIKI_PAGE_TITLES = ["Quantum computing", "Shor's algorithm", "Peter Shor", "Entanglement"]
ARXIV_PAPER_IDS = ["2208.00733v1", "quant-ph/0003151v1"]

# 2. The Wiki "allow-list" and their types from your ontology
ENTITY_MAP_WIKI = {
    # Entity Name : Ontology Class
    "Quantum computing": "QuantumConcept",
    "Qubit": "QuantumHardware",
    "Superposition": "QuantumConcept",
    "Entanglement": "QuantumConcept",
    "Shor's algorithm": "QuantumAlgorithm",
    "Grover's algorithm": "QuantumAlgorithm",
    "Quantum annealing": "QuantumConcept",
    "D-Wave Systems": "QuantumHardware",
    "Peter Shor": "Person"  # <-- UNALIGNED ENTITY
}

# 3. The arXiv "allow-list" and their types
ENTITY_MAP_ARXIV = {
    # Notice the different strings for the same concepts
    "quantum computation": "QuantumConcept",
    "quantum bit": "QuantumHardware",        # -> Aligns with 'Qubit'
    "quantum coherence": "QuantumConcept",   # -> Aligns with 'Superposition'
    "non-locality": "QuantumConcept",        # -> Aligns with 'Entanglement'
    "Shor algorithm": "QuantumAlgorithm",    # -> Aligns with "Shor's algorithm"
    "Grover algorithm": "QuantumAlgorithm",  # -> Aligns with "Grover's algorithm"
    "quantum annealer": "QuantumHardware",   # -> Aligns with 'Quantum annealing'
    "D-Wave": "QuantumHardware",             # -> Aligns with 'D-Wave Systems'
    "Lov Grover": "Person"   # <-- UNALIGNED ENTITY
}


# --- knowledge graph configuration ---

NS_RAW = Namespace("http://example.org/raw/")
REL_GENERIC = NS_RAW.is_related_to

MAX_UNPRUNED_TRIPLES = 100