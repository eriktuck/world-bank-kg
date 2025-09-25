from typing import List, Tuple, Dict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler


"""
Requirements
- extract named entities of desired type
"""

class EntityExtractor:
    def __init__(self):
        pass
        
    def extract(self, nlp, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        doc = nlp(text)

        return [(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in doc.ents]
    

