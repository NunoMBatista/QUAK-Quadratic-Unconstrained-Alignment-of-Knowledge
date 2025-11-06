from pathlib import Path

# the project root is two directories above the config file
PROJECT_ROOT = Path(__file__).parent.parent


# protege files
PROTEGE_DIR = PROJECT_ROOT / 'protege'
# specific ontology file
ONTOLOGY_FILE = PROTEGE_DIR / 'ontology.ttl'


# pipeline outputs
OUTPUT_DIR = PROJECT_ROOT / 'output'
ONTOLOGY_DIR = OUTPUT_DIR / 'ontologies'


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