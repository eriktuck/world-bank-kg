import logging
from typing import Dict, List, Any

import spacy
from spacy.pipeline import EntityRuler
from scispacy.abbreviation import AbbreviationDetector  # used to add "abbreviation_detector" in nlp pipeline

from src.acronyms import AcronymExtractor
from src.ner import EntityExtractor
from src.linker import Wikifier 
from src.reader import Reader
from src.summarize import OllamaClient

logger = logging.getLogger(__name__)


class DocumentPipeline:
    def __init__(self, file_id: str, model: str = "en_core_web_sm"):
        self.file_id = file_id
        self.nlp = spacy.load(model)
        
        # Add abbreviation detector
        if "abbreviation_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("abbreviation_detector", first=True)

        # Add entity ruler (after NER)
        if "entity_ruler" not in self.nlp.pipe_names:
            self.entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            self.entity_ruler = self.nlp.get_pipe("entity_ruler")
        
        self.acronym_extractor = AcronymExtractor(
            file_id, 
            client=OllamaClient(model="llama3.2:latest"),
            backend='ollama')
        self.entity_extractor = EntityExtractor()
        
        self.reader = Reader()  # TODO: use ArtifactStore from storage.py instead


    def process(self, md_text: str) -> Dict[str, Any]:
        logger.info("Processing document...")
        
        doc = self.nlp(md_text)
        
        acronyms = self.acronym_extractor.extract(doc)
        logger.info(f"Extracted {len(acronyms)} acronyms: {list(acronyms.keys())}")

        self.entity_extractor.add_acronym_patterns(self.entity_ruler, acronyms)

        doc = self.entity_extractor.apply_entity_ruler(self.entity_ruler, doc)

        entities = self.entity_extractor.collect_entities(doc)
        logger.info(f"Collected {len(entities)} total entities.")

        wikifier = Wikifier()
        linked_entities = wikifier.wikify(entities)

        cleaned_entities = self.entity_extractor._normalize_entities(linked_entities)

        return acronyms, cleaned_entities
    

    def run(self):
        md_text = self.reader.get_markdown(self.file_id)
        acronyms, entities = self.process(md_text)

        return {
            "doc_id": self.file_id,
            "acronyms": acronyms,
            "entities": entities,
        }
