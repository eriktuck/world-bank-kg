from typing import List, Tuple, Dict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler

"""
Requirements
- extract named entities of desired type
"""

class EntityExtractor:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        
    def extract(self, text: str, acronyms: Dict) -> List[Tuple[str, str, Tuple[int, int]]]:
        ruler: EntityRuler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "ACRONYM", "pattern": [{"LOWER": abbr.lower()}], "id": abbr}
            for abbr in acronyms
        ]
        ruler.add_patterns(patterns)

        doc = self.nlp(text)

        return [(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in doc.ents]
