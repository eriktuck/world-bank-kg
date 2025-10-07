import logging
from pathlib import Path
import json

from src.graph import KnowledgeGraph
from src.pipeline import DocumentPipeline
from src.reader import Reader
from src.storage import add_file, enrich_document_chunks, reset_storage
from src.summarize import OllamaClient, Summarizer

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logging.getLogger("__main__").setLevel(logging.DEBUG)  # for main.py
logging.getLogger("src").setLevel(logging.DEBUG)

def main():
    # Load or build the KG
    logger.info("Building Knowledge Graph...")

    kg = KnowledgeGraph.load_or_build('world-bank-kg.ttl', rebuild=True)

    # Get list of docs to process
    doc_ids = kg.get_document_ids()

    logger.debug(f"Documents in KG: {doc_ids}")
    
    reader = Reader(
        output_dir="output", 
        f_dump_middle_json=False,
        f_dump_model_output=False
    )
    
    for doc_id in doc_ids:
        doc_id = '10170637'  # TODO
        url = kg.get_url_by_id(doc_id)

        if not url:
            logger.warning(f"No URL for doc {doc_id}")
            continue 
        
        # Convert PDF to structured output
        logger.info(f'Fetching doc {doc_id} at {url}')
        # reader.process_doc(url, doc_id, lang="English")

        # Chunk and store file
        json_file_path = Path(f'output/{doc_id}/auto/{doc_id}_content_list.json')
        reset_storage()
        add_file(json_file_path, kg_id=doc_id)
    
        # Run pipeline
        pipeline = DocumentPipeline(doc_id)
        results = pipeline.run()

        # Add entities
        entities = results['entities']
        kg.add_entities(doc_id, entities)
        
        # Save updated knowledge graph
        kg.save()

        # Enrich document chunks
        logger.info("Enriching document chunks...")
        acronyms = results['acronyms']
        enrich_document_chunks(doc_id, acronyms=acronyms, entities=entities)

        logger.info(f"Pipeline ran without errors. See logs at {LOG_FILE}")

        # TODO: make dir for each file
        results_file = LOG_DIR / "pipeline_results.json"
        results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info(f"Saved results to {results_file}")

        kg.add_text_chunks(doc_id=doc_id)

        client = OllamaClient(model="llama3.2:latest")
        summarizer = Summarizer(kg, client, backend='ollama')
        chunk_graph = summarizer.build_chunk_graph()
        chunk_to_comm, hc = summarizer.detect_communities_hierarchical_leiden(chunk_graph)
        summarizer.add_communities_to_graph(chunk_to_comm)
        summarizer.summarize_communities(chunk_to_comm)

        break   # TODO


if __name__ == "__main__":
    main()
