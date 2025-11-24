from typing import List, Tuple, Dict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler
import logging
from rdflib import URIRef
import re
from urllib.parse import quote
import json

from src.linker import Wikifier

logger = logging.getLogger(__name__)

"""
Requirements
- extract named entities of desired type
"""

EXCLUDED_ENTS = ['DATE', 'TIME', 'PERCENT',
                'MONEY', 'QUANTITY', 'ORDINAL',
                'CARDINAL', 'PERSON']

class EntityExtractor:
    def __init__(self):
        pass

    
    def _sanitize_for_rdflib(self, entity: str) -> str | None:
        """Generate a URI-safe identifier for RDFLib/Turtle serialization."""
        if not entity:
            return None

        # --- URI-safe transformation ---
        safe_entity = quote(entity.replace(" ", "_"))[:100]  # limit length

        # Validate it is serializable in Turtle
        try:
            URIRef(f"http://worldbank.example.org/entity/{safe_entity}").n3()
        except Exception:
            return None

        return safe_entity
    

    def _normalize_entities(self, entities: List[List[str]]) -> List[Dict]:
        for ent in entities:
            if ent['qid']:
                rdf_safe_id = ent['qid']
            else:
                rdf_safe_id = self._sanitize_for_rdflib(ent['surface'])
            if rdf_safe_id:  # skip if invalid
                ent['rdf_safe'] = rdf_safe_id
        return entities
    

    def add_acronym_patterns(self, entity_ruler, acronyms):
        """
        Add both acronym (e.g., 'STEP') and expanded form (e.g.,
        'Systematic Tracking and Exchanges in Procurement') patterns to the EntityRuler.
        Gives them priority over SpaCy NER and allows case-insensitive matching.
        """
        patterns = []

        for abbr, expanded in acronyms.items():
            # Add abbreviation pattern
            patterns.append({
                "label": "ACRONYM",
                "pattern": [{"LOWER": abbr.lower()}],
                "id": abbr
            })

            # Add expanded form pattern (case-insensitive token sequence)
            if expanded and isinstance(expanded, str):
                tokens = [{"LOWER": t.lower()} for t in expanded.split()]
                patterns.append({
                    "label": "ACRONYM_EXPANDED",
                    "pattern": tokens,
                    "id": expanded
                })

        if patterns:
            entity_ruler.add_patterns(patterns)
            logger.debug(f"Added {len(patterns)} acronym + expanded patterns to EntityRuler.")


    def apply_entity_ruler(self, entity_ruler, doc):
        """Apply just the EntityRuler to the existing Doc."""
        return entity_ruler(doc)
    

    def collect_entities(self, doc) -> List[Dict]:
        """Collect all entity spans (including those added by EntityRuler)."""
        return [{"surface": ent.text, "label": ent.label_}
            for ent in doc.ents if ent.label_ not in EXCLUDED_ENTS]
    

def main():
    model = "en_core_web_sm"
    md_text = "This chapter introduces newly developed concepts about sustainable hazardous waste management and treatment and describes the key elements of sustainable hazardous waste management in the context of current broader issues (e.g., renewable energy and climate change). It also discusses fundamentals and basic components of sustainable hazardous waste and presents technical options for sustainable hazardous waste treatment and remediation. Microorganisms play an important role in wastewater treatment because of their immense potential for immobilization and bio-accumulative properties. Next, the chapter explains disposal of hazardous waste and reviews adjustments to meet global challenges. The choice of disposal should be based on evaluation of economics and potential pollution risks. Finally, the chapter identifies future trends and challenges for sustainable hazardous waste management/treatment, providing research recommendations to help achieve the broader goals of sustainability."
    nlp = spacy.load(model)
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
    
    with open('cache/unbis_vocab.json', 'r', encoding='utf-8') as f:
        unbis_terms = json.loads(f.read())

    patterns = [
        {
            "label": "UNBIS_TERM", 
            "pattern": term.lower(),
            "id": href
        }
        for term, href in unbis_terms.items()
    ]
    ruler.add_patterns(patterns)
    doc = nlp(md_text)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.ent_id_)

if __name__ == "__main__":
    main()
