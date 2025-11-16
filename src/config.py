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

ENTITIES_DIR = OUTPUT_DIR / 'entities'
WIKI_ENTITIES_DIR = ENTITIES_DIR / 'wiki_entities'
ARXIV_ENTITIES_DIR = ENTITIES_DIR / 'arxiv_entities'
WIKI_ENTITIES_FILE = WIKI_ENTITIES_DIR / 'entities.txt'
ARXIV_ENTITIES_FILE = ARXIV_ENTITIES_DIR / 'entities.txt'
WIKI_ENTITIES_METADATA_FILE = WIKI_ENTITIES_DIR / 'entities_metadata.json'
ARXIV_ENTITIES_METADATA_FILE = ARXIV_ENTITIES_DIR / 'entities_metadata.json'

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
ALIGNED_KG_HTML = KG_DIR / 'aligned_kg.html'

ALIGNMENTS_DIR = OUTPUT_DIR / 'alignments'
ALIGNED_ENTITIES_ANNEALER_CSV = ALIGNMENTS_DIR / 'alignment_annealer.csv'
ALIGNED_ENTITIES_NN_CSV = ALIGNMENTS_DIR / 'alignment_nn.csv'
# Backwards compatibility for older imports
ALIGNED_ENTITIES_CSV = ALIGNED_ENTITIES_ANNEALER_CSV

QUBO_OUTPUT_DIR = OUTPUT_DIR / 'qubo'
QUBO_MATRIX_CSV = QUBO_OUTPUT_DIR / 'qubo_matrix.csv'
QUBO_NODE_MATRIX_CSV = QUBO_OUTPUT_DIR / 'qubo_matrix_H_node.csv'
QUBO_STRUCTURE_MATRIX_CSV = QUBO_OUTPUT_DIR / 'qubo_matrix_H_structure.csv'
QUBO_PENALTY_MATRIX_CSV = QUBO_OUTPUT_DIR / 'qubo_matrix_H_penalty.csv'
# embedding outputs
EMBEDDINGS_DIR = OUTPUT_DIR / 'embeddings'
ENTITY_EMBEDDINGS_WIKI_PATH = EMBEDDINGS_DIR / 'entity_embeddings_wiki.pt'
ENTITY_EMBEDDINGS_ARXIV_PATH = EMBEDDINGS_DIR / 'entity_embeddings_arxiv.pt'
REL_EMBEDDINGS_PATH = EMBEDDINGS_DIR / 'relation_embeddings.npz' # Output file for H_structure


# ===================== NLP / LLM SETTINGS =====================

KG_CONSTRUCTION_MODE = "llm"  # either 'nlp' or 'llm'

NLP_MODEL = "en_core_sci_scibert"
NLP_PIPELINE = "src/kg_construction/nlp_pipeline.py"
SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_cased"

DOTENV_PATH = PROJECT_ROOT / ".env"

LLM_VENDOR = "groq"
LLM_MODEL_NAME = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.15
LLM_MAX_OUTPUT_TOKENS = 8192
LLM_CHUNK_CHAR_LIMIT = 3500
LLM_MAX_TRIPLES_PER_CHUNK = 20
LLM_MAX_ENTITIES_PER_CHUNK = 25

LLM_EXTRACTION_SYSTEM_PROMPT = (
    "You are an ontology engineer who writes compact knowledge graphs. "
    "Return valid JSON only—no code fences or commentary. "
    "Follow the configured JSON schema exactly and always emit both top-level arrays, even if they are empty."
)

LLM_EXTRACTION_USER_PROMPT = """
Extract entities and factual triples from the provided scientific text.
- Focus on quantum computing, algorithms, hardware, complexity, people, and institutions.
- Keep every surface form you encounter; duplicates will be resolved later.
- Relations should be concise verbs or predicates in snake_case (e.g., derived_from).
- If a statement is speculative, mark confidence below 0.5.
- Always return JSON with both top-level keys `entities` and `triples`. Use empty arrays when you have nothing to add.

Return only JSON that conforms to the configured schema.

Text to analyze:
{text}
"""


LLM_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "Concept",
                            "Algorithm",
                            "Hardware",
                            "Institution",
                            "Metric",
                            "Person",
                            "Other"
                        ]
                    },
                    "description": {"type": "string"}
                },
                "required": ["name"]
            }
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["subject", "relation", "object"]
            }
        }
    },
    "required": ["entities", "triples"]
}

LLM_RECON_SYSTEM_PROMPT = (
    "You are a precise data steward who canonicalizes entity names. "
    "Return valid JSON only—no code fences or commentary. "
    "Follow the configured JSON schema exactly."
)

LLM_RECON_USER_PROMPT = """
You will be given a list of entity surface forms extracted from related documents. Group
aliases that clearly refer to the same real-world concept or person, and choose the most
complete surface form as the canonical label.

Entities:
{entity_names}

Return only JSON that adheres to the configured schema.
"""

LLM_RECON_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "canonical_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical": {"type": "string"},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["canonical"]
            }
        }
    },
    "required": ["canonical_entities"]
}

LLM_RECON_MAX_NAMES = 200


# ===================== DATA FETCHING SETTINGS =====================

# this defines if the data fetchers will search for the best matches or follow a pre-defined list of articles
QUERY_MODE = 'fetch' # can be either 'search' or 'fetch'

# the query for Wikipedia and arXiv
# used for the search mode
DOMAIN_QUERY = "Quantum Algorithms"

# how many articles to fetch in the search mode
NUM_ARTICLES = 10

# these are both used for the fetch query mode
# WIKI_PAGE_TITLES = ['Quantum computing', 'Superconducting quantum computing', 'Timeline of quantum computing and communication', 'Trapped-ion quantum computer', 'Glossary of quantum computing', 'List of companies involved in quantum computing, communication or sensing', 'Rigetti Computing', 'Institute for Quantum Computing', 'Silicon Quantum Computing', 'Post-quantum cryptography']
#ARXIV_PAPER_IDS = ['2208.00733v1', 'quant-ph/0003151v1', '1311.4939v1', '1210.0736v1', '1610.02500v1', '2410.00917v1', '2210.02886v1', '0804.3401v1', 'quant-ph/0201082v1', '2410.00916v1']

WIKI_PAGE_TITLES = [
    'Quantum algorithm', 
    'Post-quantum cryptography', 
    'Quantum optimization algorithms', 
    'Quantum computing', 
    "Shor's algorithm", 
    "Grover's algorithm", 
    'Noisy intermediate-scale quantum computing', 
    'Quantum machine learning', 
    'Quantum counting algorithm', 
    'Quantum phase estimation algorithm'
]


ARXIV_PAPER_IDS = [
    '2310.03011v2', # quantum algorithms survey
    '2406.13258v3', # applications of PQC
    '2312.13636v3', # quantum optimization in operations research
    '0708.0261v1', # intro to quantum computing
    'quant-ph/9508027v2', # shor's algorithm
    '2108.10854v2', # grover's algorithm revisited
    '1801.00862v3', # NISQ
    '1611.09347v2', # QML
    'quant-ph/9805082v1', # quantum counting
    'quant-ph/9511026v1' # quantum phase estimation
]



#WIKI_PAGE_TITLES = ['Quantum computing', 'Superconducting quantum computing', 'Timeline of quantum computing and communication', 'Trapped-ion quantum computer', 'Glossary of quantum computing', 'Rigetti Computing', 'Silicon Quantum Computing', 'Quantum error correction', 'List of companies involved in quantum computing, communication or sensing', 'Institute for Quantum Computing', 'Linear optical quantum computing', "India's quantum computer", 'Post-quantum cryptography', 'Adiabatic quantum computation', 'Quantum logic gate', 'QuEra Computing Inc.', 'Topological quantum computer', 'Quantum algorithm', 'List of quantum processors', 'Quantum information science', 'Quantinuum', 'Qubit', 'List of quantum computing journals', 'Noisy intermediate-scale quantum computing', 'One-way quantum computer', 'Quantum supremacy', 'Mike Lazaridis', "Shor's algorithm", 'Elham Kashefi', "Grover's algorithm", 'D-Wave Systems', 'Quantum computing scaling laws', 'Multiverse Computing', 'Unconventional computing', 'Computing', 'Quantum memory', 'Cloud-based quantum computing', 'Willow processor', 'Reservoir computing', 'Reversible computing', 'Xanadu Quantum Technologies', 'Quantum circuit', 'Key size', 'IonQ', 'PsiQuantum', 'Nuclear magnetic resonance quantum computer', 'Quantum Artificial Intelligence Lab', 'Quantum decoherence', 'Microsoft Azure Quantum', 'Quantum Computing Since Democritus', 'Quantum engineering', 'Quantum Computation and Quantum Information', 'Jay Gambetta', 'Quantum programming', 'Krysta Svore', 'Applications of quantum mechanics', 'Qiskit', 'Quantum network', 'Atom Computing', 'Neuromorphic computing', 'List of quantum logic gates', 'Quantum Computing: A Gentle Introduction', 'Quantum annealing', 'Guillaume Verdon', 'Ancilla bit', 'Quantum machine learning', 'Raymond Laflamme', 'Threshold theorem', 'Sergio Boixo', 'IQM Quantum Computers', 'Amaravati Quantum Valley', 'Scott Aaronson', 'National Quantum Initiative Act', 'Quantum complexity theory', 'Observer effect (physics)', 'Juani Bermejo Vega', 'Quantum geometry', 'QC Ware', 'Harvest now, decrypt later', 'National Quantum Computing Centre', 'Concurrence (quantum computing)', 'Nagendra Nagaraja', 'IBM Quantum Platform', 'Quantum superposition', 'John M. Martinis', 'Alice & Bob (company)', 'Alán Aspuru-Guzik', 'Bob Coecke', 'Quantum Fourier transform', 'Majorana 1', 'Harry Buhrman', 'Technology Innovation Institute', 'Optical computing', 'Chetan Nayak', 'Microsoft Research', 'Román Orús', 'Quantum Turing machine', 'Quantum Supremacy', 'Ilyas Khan', 'Quantum neural network']
#ARXIV_PAPER_IDS = ['2211.02350v1', '2403.02240v5', '2506.15909v1', '2312.06975v1', '2302.12119v1', 'quant-ph/9701001v1', '1801.00862v3', 'quant-ph/0402010v1', 'quant-ph/0501046v1', '1804.03719v3', '2408.03613v1', '2103.07934v2', '2011.03031v2', '2212.12078v2', 'quant-ph/9505018v1', '1506.09091v3', '2403.04006v3', 'quant-ph/9505016v1', '1109.5549v1', '1811.09849v1', '1612.08091v2', '2105.04649v3', '2111.00117v3', '1401.6658v1', '1106.5712v1', '2312.07840v1', '2306.07342v4', '2307.02593v2', '2308.00583v2', '2108.07027v1', '2312.03653v1', '2112.04501v3', 'quant-ph/9804038v1', '2207.01964v4', '2003.11810v4', '2102.04452v1', '1501.00011v1', '2212.10990v1', 'quant-ph/0402090v1', '2311.05605v5', '2008.06812v2', 'quant-ph/0211158v1', '2009.13865v5', '2401.04271v4', '2102.04459v1', '2210.10776v3', '2106.07077v1', '2208.02645v2', '2308.14239v2', '2209.05469v3', 'quant-ph/0207059v1', '2403.12691v2', '1611.00664v2', '1807.07112v3', '2104.00572v3', '1508.03695v1', '2311.08445v3', '2001.09161v4', '1101.4722v3', '2405.09115v1', 'quant-ph/0008112v1', '2307.14308v1', '2005.03791v2', '1905.02666v5', '2107.08049v1', '2203.17181v1', '2303.05533v3', '2308.11740v1', '2207.01005v3', '1906.04410v2', '2110.12318v3', '1309.7650v3', 'quant-ph/0504100v1', '2009.06551v3', 'quant-ph/9802007v1', '2101.05560v1', '2507.17712v1', '2310.08437v2', '2209.11322v3', '2208.10342v2', 'quant-ph/0401019v3', '2412.05997v3', 'quant-ph/0411074v1', '2405.00304v3', '2207.10592v2', 'quant-ph/9809038v1', '2207.00021v3', '1804.03159v2', '2501.11119v1', '2112.14193v3', '2306.14948v2', '2302.02454v4', '1509.09180v4', '2110.10587v3', '2103.04502v2', '1705.00365v2', '1708.09757v2', '1204.2907v1', '2112.00984v1', '2311.13644v2']
# ===================== ONTOLOGY SETTINGS =====================

RELATION_LABELS = ["developedBy", "usesConcept", "implements"]

# the Wiki "allow-list" and their types from the ontology
ENTITY_MAP_WIKI = {
    "Grover's algorithm": "QuantumAlgorithm",
    "Shor's algorithm": "QuantumAlgorithm",
    "qubit": "QuantumConcept",
    "Noisy intermediate-scale quantum NISQ computing": "QuantumConcept",
    "Post-quantum cryptography": "QuantumConcept",
    "quantum machine learning": "QuantumConcept",
    "quantum computer technology": "QuantumHardware",
    "Quantum optimization algorithms": "QuantumAlgorithm",
    "Peter Shor": "Person",
}

ENTITY_MAP_ARXIV = {
    "Grover algorithm": "QuantumAlgorithm",
    "Shor's quantum algorithms": "QuantumAlgorithm",
    "quantum machine learning": "QuantumConcept",
    "NISQ devices": "QuantumConcept",
    "post-quantum cryptography": "QuantumConcept",
    "quantum computing": "QuantumConcept",
    "qubits": "QuantumConcept",
    "Deutsch's algorithm": "QuantumAlgorithm",
    "QOA": "QuantumAlgorithm"
}


# ===================== KNOWLEDGE GRAPH SETTINGS =====================


NS_RAW = Namespace("http://unpruned.local/") # namespace for raw relations extracted from text
REL_GENERIC = NS_RAW.is_related_to # generic relation for untyped relations

MAX_UNPRUNED_TRIPLES = None # max number of triples in the unpruned KG before applying pruning

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
EPOCHS = 500      # number of training loops
LEARNING_RATE = 0.001
GAE_DROPOUT = 0.2 # dropout used inside the graph encoders

GAEA_MMD_WEIGHT = 0.5       # weight applied to the distribution alignment term
GAEA_STATS_WEIGHT = 0.1     # weight applied to first/second moment alignment
GAEA_MAX_ALIGN_SAMPLES = 2048 # max samples drawn when estimating MMD

USE_SCIBERT_FEATURES = True # whether to use SciBERT features for node attributes (false to set the initial features to identity matrix)

# toggle to fully skip the graph autoencoder (GAE/GAEA) stage and reuse the raw
# node features (e.g., SciBERT) as the exported entity embeddings. Set to False
# when you only need surface-form embeddings and want to avoid GNN training time.
USE_GAE_FOR_ENTITY_EMBEDDINGS = True

# optionally enforce soft constraints between known aligned entities while
# training the joint graph autoencoder. Each pair should specify the canonical
# Wiki-side label and the ArXiv-side label (matching the canonicalized entity
# names saved in the KG step). When disabled or when the list is empty, the
# model falls back to unsupervised joint training.
USE_ANCHOR_ALIGNMENTS = True
ANCHOR_ALIGNMENT_WEIGHT = 1.0
ANCHOR_ALIGNMENT_PAIRS = [
    # example:
    # {"wiki": "Grover's algorithm", "arxiv": "Grover algorithm"},
    {"wiki": "qubit", "arxiv": "qubits"},
    {"wiki": "Grover's algorithm", "arxiv": "Grover algorithm"}
]


# ===================== ANNEALING SETTINGS =====================

QUBO_NODE_WEIGHT = 1.0
QUBO_STRUCTURE_WEIGHT = 0.2
QUBO_WIKI_PENALTY = 2.0
QUBO_ARXIV_PENALTY = 2.0
ANNEALER_NUM_READS = 100
ANNEALER_BETA_RANGE = None  # e.g., (0.1, 4.0) for custom cooling schedule
ANNEALER_SEED = None
ANNEALER_MAX_STRUCTURAL_PAIRS = 2000

# ===================== ALIGNMENT SOLVER SETTINGS =====================

DEFAULT_SIMILARITY_THRESHOLD = 0.75