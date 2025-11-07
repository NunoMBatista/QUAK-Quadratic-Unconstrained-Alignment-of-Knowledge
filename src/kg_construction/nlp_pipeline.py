import spacy
from nltk.tokenize import sent_tokenize
import nltk
from src.config import *
import warnings

# ignore annoying warnings from spaCy about model version mismatches
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class NLPPipeline:
    """
    uses SciBERT (from the spaCy library) to perform NER.

    this will find all entities that the model knows, not just the filtered list.

    uses simple co-occurrence for RE.
    it just checks if two entities are in the same sentence to determine if
    the triple (entity_a, "is_related_to", entity_b) exists.
    we will then use our ontology to determine what "is_related_to" means.
    """
    
            
    def __init__(self):
        """
        this loads the 'en_core_sci_scibert' model, which is SciBERT
        fine-tuned for NER on scientific text.
        
        it also downloads the tokenizer, if it does not exist.
        """
        
        # download the sentence tokenizer model (punkt) if it doesn't exist
        self._load_tokenizer_model()
        
        # load the SciBERT NER model
        self._load_nlp_model()
       

    def _load_nlp_model(self):
        """loads the SciBERT NER model from spaCy"""
        try:
            self.nlp = spacy.load(NLP_MODEL)
            print("-> SciBERT NER model loaded.")
        except IOError:
            print(f"-> '{NLP_MODEL}' model not found.")
            print(f"    -> run: pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/{NLP_MODEL}-0.5.3.tar.gz")
            exit(1)            
            

    def _load_tokenizer_model(self):
        """downloads punkt if it does not exist yet"""
        
        """
        NOTE: IF THE DOWNLOAD FAILS, YOU NEED TO 
        MANUALLY REMOVE THE NLTK DATA FOLDER
        
        rm ~/nltk_data/tokenizers/punkt.zip
        """
        try:
            nltk.data.find('tokenizers/punkt')
            print("-> NLTK 'punkt' tokenizer model found.")
        except LookupError:
            print("-> NLTK 'punkt' tokenizer not found. Downloading...")
            nltk.download('punkt')
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
            print("-> NLTK 'punkt_tab' tokenizer model found.")
        except LookupError:
            print("-> NLTK 'punkt_tab' tokenizer not found. Downloading...")
            nltk.download('punkt_tab')
            

    def extract_triples(self, text):
        """
        returns a list of raw triples found by the NER model.
        
        Parameters
        ----------
        text : str
            The raw text of an article/abstract.
            
        Returns
        -------
        list of tuples
            A list of (subject_text, "related_to", object_text) triples.
        """

        # solve type hinting stuff
        self.nlp: spacy.language.Language

        triples = []
        
        # split the text into sentences
        sentences = sent_tokenize(text)

        for sentence in sentences:
            doc = self.nlp(sentence)
            
            # get the entities found in this specific sentence
            entities_in_sentence = []
            for ent in doc.ents:
                entities_in_sentence.append(ent.text)

            # if 2+ entities co-occur in the same sentence, create relations
            if len(entities_in_sentence) >= 2:
                # this creates all pairs of co-occurring entities
                for i in range(len(entities_in_sentence)):
                    for j in range(i + 1, len(entities_in_sentence)):
                        e1 = entities_in_sentence[i]
                        e2 = entities_in_sentence[j]
                        
                        # avoid creating triples with the same entity
                        if e1.lower() != e2.lower():
                            triples.append((e1, "related_to", e2))
                            triples.append((e2, "related_to", e1))
                            # i'm adding both directions for now (this will be rectified in the ontology mapping step)

        return triples


# --- test the file directly ---
if __name__ == "__main__":
    test_text = """
    Quantum computing is a new paradigm. It relies on the Qubit.
    Shor's algorithm is a famous quantum algorithm. 
    Another key concept is Entanglement.
    Peter Shor developed Shor's algorithm.
    """
    
    pipeline = NLPPipeline()
    
    print(f"\n--- extracting raw triples from test text ---")
    triples = pipeline.extract_triples(test_text)
    
    for t in triples:
        print(t)