import spacy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple, Union
from spacy.tokens import Doc, Span, Token

from nltk.tokenize import sent_tokenize
import nltk
from src.config import *
import warnings

# ignore annoying warnings from spaCy about model version mismatches
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SUBJECT_DEPS: Set[str] = {"nsubj", "nsubjpass"}
OBJECT_DEPS: Set[str] = {"dobj", "obj", "pobj", "attr", "dative", "oprd"}
COPULAR_DEPS: Set[str] = {"attr", "acomp", "oprd"}
ATTRIBUTE_COMPLEMENT_DEPS: Set[str] = set(COPULAR_DEPS)
APPOSITIVE_DEPS: Set[str] = {"appos"}


@dataclass(frozen=True)
class RelationTriple:
    subject: str
    relation: str
    object: str


class NLPPipeline:
    """SciSpaCy-powered NER with lightweight dependency-based relation extraction."""
    
            
    def __init__(self):
        """
        this loads the 'en_core_sci_scibert' model, which is SciBERT
        fine-tuned for NER on scientific text.
        
        it also downloads the tokenizer, if it does not exist.
        """
        
        # this will hold the loaded SciBERT model
        self.nlp: spacy.language.Language
        
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
            

    def extract_triples(
        self,
        text: str,
        *,
        return_entities: bool = False,
    ) -> Union[List[RelationTriple], Tuple[List[RelationTriple], List[str]]]:
        """
        extract relations using dependency patterns."
        
        Parameters
        ----------
        text : str
            The raw text to process.
        return_entities : bool, optional
            When True, also returns the unique entities detected by the NER model.

        Returns
        -------
        List[RelationTriple] or Tuple[List[RelationTriple], List[str]]
            The extracted relation triples. Optionally returns the unique entity names.
        
        """

        self.nlp: spacy.language.Language
        
        #sentences = [sent.strip() for sent in sent_tokenize(text) if sent.strip()]
        
        # sent_tokenize returns a list of sentences from the raw text.
        sentences = []
        for sent in sent_tokenize(text):
            sent_clean = sent.strip() # remove leading/trailing whitespace
            if sent_clean:
                sentences.append(sent_clean)
        
        if not sentences:
            return ([], []) if return_entities else []

        triples: Set[RelationTriple] = set()
        entity_lookup: Dict[str, str] = {}
        # a doc is a processed spaCy object
        for doc in self.nlp.pipe(sentences, batch_size=8):
            if return_entities:
                for ent in doc.ents:
                    ent_text = ent.text.strip()
                    if not ent_text:
                        continue
                    key = ent_text.lower()
                    if key not in entity_lookup:
                        entity_lookup[key] = ent_text
            
            # build a mapping from token index to entity span
            token_to_ent = self._build_token_entity_map(doc)
            if not token_to_ent:
                continue

            # extract relations based on verbs and add them to the set
            triples.update(self._extract_verb_relations(doc, token_to_ent))
            
            # extract copular relations and add them to the set
            triples.update(self._extract_copular_relations(doc, token_to_ent))

            # extract attribute complements and appositives for richer coverage
            triples.update(self._extract_attribute_relations(doc, token_to_ent))
            triples.update(self._extract_appositive_relations(doc, token_to_ent))
        sorted_triples = sorted(
            triples,
            key=lambda t: (t.subject.lower(), t.relation.lower(), t.object.lower()),
        )

        if not return_entities:
            return sorted_triples

        sorted_entities = sorted(entity_lookup.values(), key=lambda value: value.lower())
        return sorted_triples, sorted_entities

    @staticmethod
    def _build_token_entity_map(doc: Doc) -> Dict[int, Span]:
        mapping: Dict[int, Span] = {}
        for ent in doc.ents:
            for token in ent:
                mapping[token.i] = ent
        return mapping

    def _extract_verb_relations(
        self,
        doc: Doc,
        token_to_ent: Dict[int, Span],
    ) -> Set[RelationTriple]:
        triples: Set[RelationTriple] = set()
        """
        extracts relations based on the verbs in the sentence
        and their subject/object dependencies.

        Parameters
        ----------
        doc : Doc
            The spaCy processed document.
        token_to_ent : Dict[int, Span]
            Mapping from token indices to entity spans.
            
        Returns
        -------
        Set[RelationTriple]
            A set of extracted relation triples.
        """

        for token in doc:
            if token.pos_ != "VERB":
                continue

            subjects = self._collect_entities(token.children, SUBJECT_DEPS, token_to_ent)
            objects = self._collect_entities(token.children, OBJECT_DEPS, token_to_ent)

            # handle prepositional attachments on the verb
            prep_objects = self._collect_prep_objects(token.children, token_to_ent)
            if prep_objects:
                objects = objects + prep_objects

            if not subjects or not objects:
                continue

            relation = self._compose_relation_label(token)
            for subj in subjects:
                for obj in objects:
                    if subj.start == obj.start and subj.end == obj.end:
                        continue
                    triples.add(RelationTriple(subj.text.strip(), relation, obj.text.strip()))

        return triples

    def _extract_copular_relations(
        self,
        doc: Doc,
        token_to_ent: Dict[int, Span],
    ) -> Set[RelationTriple]:
        triples: Set[RelationTriple] = set()
        """
        extracts copular relations (e.g., "X is a Y") from the document.
        Parameters
        ----------
        doc : Doc
            The spaCy processed document.
        token_to_ent : Dict[int, Span]
            Mapping from token indices to entity spans.
            
        Returns
        -------
        Set[RelationTriple]
            A set of extracted copular relation triples.
        """


        for token in doc:
            if not any(child.dep_ == "cop" for child in token.children):
                continue

            subjects = self._collect_entities(token.children, SUBJECT_DEPS, token_to_ent)
            complements = self._entities_for_token(token, token_to_ent)
            complements.extend(self._collect_entities(token.children, COPULAR_DEPS, token_to_ent))

            if not subjects or not complements:
                continue

            relation = "is_a"
            for subj in subjects:
                for comp in complements:
                    if subj.start == comp.start and subj.end == comp.end:
                        continue
                    triples.add(RelationTriple(subj.text.strip(), relation, comp.text.strip()))

        return triples

    def _extract_attribute_relations(
        self,
        doc: Doc,
        token_to_ent: Dict[int, Span],
    ) -> Set[RelationTriple]:
        triples: Set[RelationTriple] = set()

        for token in doc:
            if token.dep_ not in ATTRIBUTE_COMPLEMENT_DEPS:
                continue

            complements = self._entities_for_token(token, token_to_ent)
            if not complements:
                continue

            head = token.head
            subjects = self._collect_entities(head.children, SUBJECT_DEPS, token_to_ent)
            if not subjects:
                subjects = self._entities_for_token(head, token_to_ent)

            if not subjects:
                continue

            for subj in subjects:
                for comp in complements:
                    if subj.start == comp.start and subj.end == comp.end:
                        continue
                    triples.add(RelationTriple(subj.text.strip(), "is_a", comp.text.strip()))

        return triples

    def _extract_appositive_relations(
        self,
        doc: Doc,
        token_to_ent: Dict[int, Span],
    ) -> Set[RelationTriple]:
        triples: Set[RelationTriple] = set()

        for token in doc:
            if token.dep_ not in APPOSITIVE_DEPS:
                continue

            head = token.head
            head_entities = self._entities_for_token(head, token_to_ent)
            if not head_entities:
                continue

            appos_entities = self._entities_for_token(token, token_to_ent)
            if not appos_entities:
                appos_entities = self._collect_entities(token.children, SUBJECT_DEPS.union(OBJECT_DEPS), token_to_ent)

            if not appos_entities:
                continue

            for head_ent in head_entities:
                for appos_ent in appos_entities:
                    if head_ent.start == appos_ent.start and head_ent.end == appos_ent.end:
                        continue
                    triples.add(RelationTriple(head_ent.text.strip(), "is_a", appos_ent.text.strip()))

        return triples

    @staticmethod
    def _collect_entities(
        candidates: Iterable[Token],
        allowed_deps: Set[str],
        token_to_ent: Dict[int, Span],
    ) -> List[Span]:
        entities: List[Span] = []
        for child in candidates:
            if child.dep_ not in allowed_deps:
                continue
            entity = NLPPipeline._entities_for_token(child, token_to_ent)
            if entity:
                entities.extend(entity)
        return entities

    @staticmethod
    def _collect_prep_objects(
        candidates: Iterable[Token],
        token_to_ent: Dict[int, Span],
    ) -> List[Span]:
        entities: List[Span] = []
        for child in candidates:
            if child.dep_ != "prep":
                continue
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    entity = NLPPipeline._entities_for_token(grandchild, token_to_ent)
                    if entity:
                        entities.extend(entity)
        return entities

    @staticmethod
    def _entities_for_token(
        token: Token,
        token_to_ent: Dict[int, Span],
    ) -> List[Span]:
        entity = token_to_ent.get(token.i)
        if entity:
            return [entity]

        # walk up simple compounds to recover multi-token entities
        compounds: List[Span] = []
        for child in token.children:
            if child.dep_ == "compound" and child.i in token_to_ent:
                compounds.append(token_to_ent[child.i])
        return compounds

    @staticmethod
    def _compose_relation_label(token: Token) -> str:
        lemma = token.lemma_.lower() if token.lemma_ else token.text.lower()
        if lemma == "be":
            lemma = "is"

        parts: List[str] = [lemma]

        particles = [child.lemma_.lower() for child in token.children if child.dep_ == "prt" and child.lemma_]
        parts.extend(sorted(set(particles)))

        prepositions = []
        for child in token.children:
            if child.dep_ != "prep" or not child.lemma_:
                continue
            phrase = [child.lemma_.lower()]
            pobj = next((gc.lemma_.lower() for gc in sorted(child.children, key=lambda t: t.i) if gc.dep_ == "pobj" and gc.lemma_), None)
            if pobj:
                phrase.append(pobj)
            prepositions.append("_".join(phrase))
        parts.extend(sorted(set(prepositions)))

        relation = "_".join(filter(None, parts))
        return relation or "related_to"


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