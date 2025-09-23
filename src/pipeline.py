import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import spacy
from llama_index.core import Document, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from src.acronyms import AcronymExtractor
from src.ner import EntityExtractor
from src.storage import STORAGE_DIR

EXCLUDED_ENTS = ['DATE', 'TIME', 'PERCENT',
                'MONEY', 'QUANTITY', 'ORDINAL',
                'CARDINAL', 'PERSON']
DOC_ID = "237200cd-911a-40bf-b84b-2b5e94afeb2b"


LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if main() is called multiple times
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def load_document_from_storage(storage_dir: str, doc_id: str):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    docstore = storage_context.docstore
    return docstore.get_document(doc_id)

### --------------------------------------------------------------------------

class KnowledgeGraphPipeline:
    def __init__(self):
        self.acronym_extractor = AcronymExtractor()
        self.entity_extractor = EntityExtractor()
    
    def process_document(self, doc) -> Dict[str, Any]:
        logger.info("Extracting acronyms...")
        acronyms = self.acronym_extractor.extract(doc.text)
        logger.info(f"Acronyms found {acronyms}")

        raw_entities = self.entity_extractor.extract(doc.text)
        # logger.info(f"Raw entities: {[(ent[0], ent[1]) for ent in raw_entities if ent[1] not in EXCLUDED_ENTS]}")

        return {
            "doc_id": doc.doc_id,
            "acronyms": acronyms,
            "entities": raw_entities,
        }

### -------------------------------------------------------------------------

def main():
    pipeline = KnowledgeGraphPipeline()
    text = load_document_from_storage(STORAGE_DIR, DOC_ID)
    results = pipeline.process_document(text)

    logger.info(f"Pipeline ran without errors. See logs at {LOG_FILE}")

    # Save results separately as JSON file for inspection
    results_file = LOG_DIR / "pipeline_results.json"
    results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()