from typing import List, Tuple
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

"""
Requirements
- extract named entities of desired type
"""

class EntityExtractor:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        
    def extract(self, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        doc = self.nlp(text)
        return [(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in doc.ents]
