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

KG_DIR = OUTPUT_DIR / 'KGs'
KG_WIKI_UNPRUNED_PATH = KG_DIR / 'kg_wiki_unpruned.ttl'
KG_ARXIV_UNPRUNED_PATH = KG_DIR / 'kg_arxiv_unpruned.ttl'
KG_WIKI_FINAL_PATH = KG_DIR / 'kg_wiki_final.ttl'
KG_ARXIV_FINAL_PATH = KG_DIR / 'kg_arxiv_final.ttl'
KG_ALIGNED_PATH = KG_DIR / 'kg_aligned.ttl'

ALIGNMENTS_DIR = OUTPUT_DIR / 'alignments'
ALIGNED_ENTITIES_ANNEALER_CSV = ALIGNMENTS_DIR / 'alignment_annealer.csv'
ALIGNED_ENTITIES_NN_CSV = ALIGNMENTS_DIR / 'alignment_nn.csv'
# Backwards compatibility for older imports
ALIGNED_ENTITIES_CSV = ALIGNED_ENTITIES_ANNEALER_CSV


# embedding outputs
EMBEDDINGS_DIR = OUTPUT_DIR / 'embeddings'
ENTITY_EMBEDDINGS_WIKI_PATH = EMBEDDINGS_DIR / 'entity_embeddings_wiki.pt'
ENTITY_EMBEDDINGS_ARXIV_PATH = EMBEDDINGS_DIR / 'entity_embeddings_arxiv.pt'
REL_EMBEDDINGS_PATH = EMBEDDINGS_DIR / 'relation_embeddings.npz' # Output file for H_structure


# ===================== NLP SETTINGS =====================

NLP_MODEL = "en_core_sci_scibert"
NLP_PIPELINE = "src/kg_construction/nlp_pipeline.py"


# ===================== DATA FETCHING SETTINGS =====================

# this defines if the data fetchers will search for the best matches or follow a pre-defined list of articles
QUERY_MODE = 'fetch' # can be either 'search' or 'fetch'

# the query for Wikipedia and arXiv
# used for the search mode
DOMAIN_QUERY = "Quantum Computing"

# how many articles to fetch in the search mode
NUM_ARTICLES = 10

# these are both used for the fetch query mode
WIKI_PAGE_TITLES = ['Quantum computing', 'Superconducting quantum computing', 'Timeline of quantum computing and communication', 'Trapped-ion quantum computer', 'Glossary of quantum computing', 'List of companies involved in quantum computing, communication or sensing', 'Rigetti Computing', 'Institute for Quantum Computing', 'Silicon Quantum Computing', 'Post-quantum cryptography']
ARXIV_PAPER_IDS = ['2208.00733v1', 'quant-ph/0003151v1', '1311.4939v1', '1210.0736v1', '1610.02500v1', '2410.00917v1', '2210.02886v1', '0804.3401v1', 'quant-ph/0201082v1', '2410.00916v1']

# ===================== ONTOLOGY SETTINGS =====================

RELATION_LABELS = ["developedBy", "usesConcept", "implements"]

# the Wiki "allow-list" and their types from the ontology
ENTITY_MAP_WIKI = {
    # Entity Name : Ontology Class
    "features": "QuantumConcept",
    "quantum_computer": "QuantumHardware",
    "possibilities": "QuantumConcept",
    "quantum_measurements": "Person",
    "quantum": "QuantumAlgorithm",
    "computing": "QuantumConcept",
    "Peter Shor": "Person"  # <-- UNALIGNED ENTITY
}

# the arXiv "allow-list" and their types from the ontology
ENTITY_MAP_ARXIV = {
    # notice the different strings for the same concepts
    "quantum": "QuantumConcept",
    "classical": "QuantumHardware",
    "distributed": "QuantumConcept",
    "Internet": "Person",
    "computations": "Algorithm",
    "nodes": "Person",
    "Internet": "Algorithm",
    "Lov Grover": "Person"   # <-- UNALIGNED ENTITY
}


# ===================== KNOWLEDGE GRAPH SETTINGS =====================


NS_RAW = Namespace("http://example.org/raw/") # namespace for raw relations extracted from text
REL_GENERIC = NS_RAW.is_related_to # generic relation for untyped relations

MAX_UNPRUNED_TRIPLES = 200 # max number of triples in the unpruned KG before applying pruning

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


# ===================== GRAPH AUTOENCODER SETTINGS =====================

EMBEDDING_DIM = 64 # final vector size (e.g., 64)
HIDDEN_DIM = 128  # GCN hidden layer size
EPOCHS = 200      # number of training loops
LEARNING_RATE = 0.01

USE_SCIBERT_FEATURES = True # whether to use SciBERT features for node attributes (false to set the initial features to identity matrix)