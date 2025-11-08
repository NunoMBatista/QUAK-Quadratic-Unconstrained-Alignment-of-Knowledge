from pathlib import Path
from rdflib import Namespace

# ===================== VISUALIZATION SETTINGS =====================

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
KG_WIKI_FINAL_PATH = KG_DIR / 'kg_wiki_final.ttl'
KG_ARXIV_FINAL_PATH = KG_DIR / 'kg_arxiv_final.ttl'


# embedding outputs
EMBEDDINGS_DIR = OUTPUT_DIR / 'embeddings'
ENTITY_EMBEDDINGS_WIKI_PATH = EMBEDDINGS_DIR / 'entity_embeddings_wiki.pt'
ENTITY_EMBEDDINGS_ARXIV_PATH = EMBEDDINGS_DIR / 'entity_embeddings_arxiv.pt'
REL_EMBEDDINGS_PATH = EMBEDDINGS_DIR / 'relation_embeddings.npz' # Output file for H_structure


# # ===================== NLP SETTINGS =====================

NLP_MODEL = "en_core_sci_scibert"
NLP_PIPELINE = "src/kg_construction/nlp_pipeline.py"


# # ===================== DATA FETCHING SETTINGS =====================

# this defines if the data fetchers will search for the best matches or follow a pre-defined list of articles
QUERY_MODE = 'search' # can be either 'search' or 'fetch'

# the query for Wikipedia and arXiv
# used for the search mode
DOMAIN_QUERY = "Quantum Computing"

# how many articles to fetch in the search mode
NUM_ARTICLES = 10

# these are both used for the fetch query mode
WIKI_PAGE_TITLES = ["Quantum computing", "Shor's algorithm", "Peter Shor", "Entanglement"]
ARXIV_PAPER_IDS = ["2208.00733v1", "quant-ph/0003151v1"]


# # ===================== ONTOLOGY SETTINGS =====================

RELATION_LABELS = ["developedBy", "usesConcept", "implements"]

# the Wiki "allow-list" and their types from the ontology
ENTITY_MAP_WIKI = {
    # Entity Name : Ontology Class
    "features": "QuantumConcept",
    "quantum_computer": "QuantumHardware",
    "possibilities": "QuantumConcept",
    "quantum_measurements": "Person",
    "quantum": "QuantumAlgorithm",
    "computation": "QuantumConcept",
    "Shor's algorithm": "QuantumAlgorithm",
    "Grover's algorithm": "QuantumAlgorithm",
    "Quantum annealing": "QuantumConcept",
    "D-Wave Systems": "QuantumHardware",
    "Peter Shor": "Person"  # <-- UNALIGNED ENTITY
}

# the arXiv "allow-list" and their types from the ontology
ENTITY_MAP_ARXIV = {
    # notice the different strings for the same concepts
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


# # # ===================== KNOWLEDGE GRAPH SETTINGS =====================


NS_RAW = Namespace("http://example.org/raw/") # namespace for raw relations extracted from text
REL_GENERIC = NS_RAW.is_related_to # generic relation for untyped relations

MAX_UNPRUNED_TRIPLES = 1000 # max number of triples in the unpruned KG before applying pruning

# we need to put the ontology namespace here for building the KG
# we can find it in the first line of the ontology.ttl file
NS_ONT_BASE = "urn:webprotege:ontology:1d3f8981-9e47-4c05-89fc-7f94b1a68603#" 
NS_ONT = Namespace(NS_ONT_BASE)

# physics configuration for graph layout, it's used in /src/utils/graph_visualizer.py
GRAPH_PHYSICS = """
    {
        "enabled": true,
        "barnesHut": {
        "gravitationalConstant": -25000,
        "centralGravity": 1.5,
        "springLength": 400,
        "springConstant": 0.04,
        "damping": 0.09,
        "avoidOverlap": 0
        },
        "solver": "barnesHut",
        "minVelocity": 0.75
    }
"""