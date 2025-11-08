import re
from src.config import *
from src.kg_construction.fetch_data import fetch_wiki_data, fetch_arxiv_data
from src.kg_construction.nlp_pipeline import NLPPipeline
from rdflib import Graph, Namespace, RDF  # <-- IMPORT RDF
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

def prune_kgs():
    """
    loads the "messy" unpruned KGs and creates clean, pruned
    KGs based on the entity maps and ontology rules.
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
            
            # ==== Rule-Based Relation Extraction Block ====
            
            predicate = None
            # if the subject is a QuantumAlgorithm and the object is a Person, we can say 'developedBy'
            if s_type == 'QuantumAlgorithm' and o_type == 'Person':
                predicate = NS_ONT['developedBy']

            # if the subject is a QuantumHardware and the object is a QuantumConcept, we can say 'implements'
            elif s_type == 'QuantumHardware' and o_type == 'QuantumConcept':
                predicate = NS_ONT['implements']
                
            # we can try other rules
            else:
                core_types = ['QuantumAlgorithm', 'QuantumConcept', 'QuantumHardware']
                if s_type in core_types and o_type in core_types:
                    predicate = NS_ONT['usesConcept']

            # add the clean Triple
            if predicate != None:
                g_clean.add((NS_ONT[clean_for_uri(s_canon)], predicate, NS_ONT[clean_for_uri(o_canon)]))
                
    # save the final, clean graph
    print(f"    -> Saving {len(g_clean)} clean triples to {clean_path}")
    g_clean.serialize(destination=str(clean_path), format="turtle")


def build_unpruned_kgs():
    """
    this will build the unpruned KGs for both Wikipedia and
    arXiv data.
    
    it will save them to disk as turtle files, no need to return them.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    
    print("\n--- STEP 1: FETCH THE DATA ---")
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
    
    # [:MAX_UNPRUNED_TRIPLES] for quick testing
    for subj, pred, obj in wiki_triples[:MAX_UNPRUNED_TRIPLES]:
        # clean the strings before adding them
        subj_clean = clean_for_uri(subj)
        obj_clean = clean_for_uri(obj)
        
        # add the triple to the graph with the generic relation
        wiki_kg.add((NS_RAW[subj_clean], REL_GENERIC, NS_RAW[obj_clean]))
        
    for subj, pred, obj in arxiv_triples[:MAX_UNPRUNED_TRIPLES]:
        # clean the strings before adding them
        subj_clean = clean_for_uri(subj)
        obj_clean = clean_for_uri(obj)
        arxiv_kg.add((NS_RAW[subj_clean], REL_GENERIC, NS_RAW[obj_clean]))
        
    # -----------------------
    
    # save the unpruned KGs to files
    print(f"\n-> Saving {len(wiki_kg)} wiki triples to {KG_WIKI_UNPRUNED_PATH}")
    wiki_kg.serialize(destination=str(KG_WIKI_UNPRUNED_PATH), format='turtle')
    
    print(f"-> Saving {len(arxiv_kg)} arXiv triples to {KG_ARXIV_UNPRUNED_PATH}")
    arxiv_kg.serialize(destination=str(KG_ARXIV_UNPRUNED_PATH), format='turtle')

if __name__ == "__main__":
    KG_DIR.mkdir(exist_ok=True, parents=True)
    build_unpruned_kgs() # get the unpruned KGs into the disk
    prune_kgs() # prune them

    print("\n--- VISUALIZING RAW ---")
    visualize_ttl(KG_WIKI_UNPRUNED_PATH,  KG_DIR / 'unpruned_wiki_kg.html')
    visualize_ttl(KG_ARXIV_UNPRUNED_PATH, KG_DIR / 'unpruned_arxiv_kg.html')
    
    print("\n--- VISUALIZING PRUNED ---")
    visualize_ttl(KG_WIKI_FINAL_PATH,  KG_DIR / 'pruned_wiki_kg.html')
    visualize_ttl(KG_ARXIV_FINAL_PATH, KG_DIR / 'pruned_arxiv_kg.html')