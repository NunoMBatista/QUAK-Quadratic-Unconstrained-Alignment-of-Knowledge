"""
run this in the root directory with:

python3 -m src.utils.graph_visualizer
"""


from pathlib import Path
from rdflib import Graph
from pyvis.network import Network
from src.config import ONTOLOGY_FILE, ONTOLOGY_DIR, GRAPH_PHYSICS

# --- quick configuration ---

# /protege/ontology.ttl
INPUT_FILE = Path(__file__).parent.parent.parent / 'protege' / 'ontology.ttl'

# create output directory if it does not exist
ONTOLOGY_DIR.mkdir(exist_ok=True, parents=True) # /output/ontologies
# where the visualization is saved
OUTPUT_FILE = ONTOLOGY_DIR / 'ontology_visualization.html' # /output/ontologies/ontology_visualization.html

# --- end configuration ---


def clean_label(uri_or_literal):
    """
    Cleans URI or literal for better display.
    e.g., http://www.w3.org/2000/01/rdf-schema#label -> label
    
    Parameters
    ----------
    uri_or_literal : URIRef or Literal
        The URI or literal to clean.
        
    Returns
    -------
    str
        The cleaned label.
    """
    label = str(uri_or_literal)
    if '#' in label:
        return label.split('#')[-1]
    if '/' in label:
        return label.split('/')[-1]
    return label


def visualize_ttl(ttl_path, output_html_path):
    """
    Loads TTL file and generates interactive HTML visualization.
    
    Parameters
    ----------
    ttl_path : Path
        The path to the input TTL file.
    output_html_path : Path
        The path to the output HTML file.


    Returns
    -------
    None
    """
    
    print(f"loading graph from: {ttl_path}\n")
    g = Graph()
    g.parse(ttl_path, format="turtle") # load the TTL file into the graph
    
    net = Network(height="800px", width="100%", directed=True) # define an empty visualization network

    # configure physics for better layout
    all_options = f"""
    {{
    "physics": {GRAPH_PHYSICS},
    "configure": {{
        "enabled": true,
        "filter": ["physics"]
    }}
    }}
    """
    net.set_options(all_options)

    # iterate over all triples in the graph
    for s, p, o in g:
        # get clean labels for nodes and edges
        s_label = clean_label(s)
        p_label = clean_label(p)
        o_label = clean_label(o)

        # add the nodes and edge to the pyvis network
        # title provides the hover-text (the full URI)
        net.add_node(s_label, label=s_label, title=str(s))
        net.add_node(o_label, label=o_label, title=str(o))
        net.add_edge(s_label, o_label, label=p_label, title=str(p))

    # Save and show the interactive HTML file
    net.show(str(output_html_path), notebook=False)
    print(f"saved graph visualization to: {Path(output_html_path).resolve()}")


if __name__ == "__main__":
    visualize_ttl(INPUT_FILE, OUTPUT_FILE)