import shutil
import logging
from pathlib import Path
from typing import Dict, List

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

from src.parser import CustomParser

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="secrets/.env")

STORAGE_DIR = Path("./storage")
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "documents"


from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

def reset_storage():
    """Delete old storage directories for a clean state."""
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
        logger.info("Deleted old llamaindex storage.")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info("Deleted old ChromaDB storage.")


def _init_storage() -> StorageContext:
    """Initialize a storage context with docstore + Chroma vector store."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    docstore = SimpleDocumentStore()
    return StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)


def _process_file(file_path: Path, storage_context: StorageContext, parser: CustomParser, kg_id: str) -> str:
    """Parse file into a Document + Nodes, index them, and add to storage."""
    raw_text = Path(file_path).read_text()

    # Create Document
    doc = Document(text=raw_text, metadata={"source": str(file_path)}, doc_id=kg_id)

    # Parse into TextNodes
    nodes = parser.get_nodes_from_documents([doc])

    for n in nodes:
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc.doc_id)

    # Store nodes documents and nodes in doc store
    storage_context.docstore.add_documents([doc])
    storage_context.docstore.add_documents(nodes)

    # Insert into VectorStore
    VectorStoreIndex(nodes, storage_context=storage_context)

    return doc.doc_id


def add_file(file_path: str, kg_id: str):
    """
    Add a file to the docstore + Chroma vector store.
    If reset=True, delete existing stores and rebuild from scratch.
    """
    storage_context = _init_storage()
    parser = CustomParser(include_metadata=True, include_prev_next_rel=True)

    llama_id = _process_file(Path(file_path), storage_context, parser, kg_id)

    storage_context.persist(persist_dir=str(STORAGE_DIR))

    logger.info(f"Added document {llama_id} from {file_path}.")
    
    return llama_id


def load_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        persist_dir=str(STORAGE_DIR),
        vector_store=vector_store
    )
    
    index = load_index_from_storage(storage_context)
    
    return index


def load_document(storage_dir: str, doc_id: str):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    docstore = storage_context.docstore
    return docstore.get_document(doc_id)


def enrich_chunks_with_annotations(chunks: List[Dict], 
                                   acronyms: Dict[str, str], 
                                   entities: List[Dict]) -> List[Dict]:
    """
    Enrich each chunk with acronyms and entities that appear in its text.

    Parameters
    ----------
    chunks : list of dict
        Each dict should have at least {"text": "..."}.
    acronyms : dict
        Mapping like {"SEMARNAT": "Secretaria de Medio Ambiente ..."}.
    entities : list of dicts
        Each entity dict like:
          {
            "surface": "...",
            "label": "...",
            "qid": "...",
            "safe_id": "..."
          }

    Returns
    -------
    list of dict
        Same chunks, but with added keys:
        - "acronyms_found": list of acronyms present
        - "entities_found": list of entity dicts present
    """
    for chunk in chunks:
        text = chunk.get("text", "")

        # Find acronyms present
        acronyms_found = {
            acr: expansion
            for acr, expansion in acronyms.items()
            if acr in text or expansion in text
        }

        # Find entities present by surface string
        entities_found = [
            ent for ent in entities 
            if ent.get("surface") and ent["surface"] in text
        ]

        chunk["acronyms_found"] = acronyms_found
        chunk["entities_found"] = entities_found

        logger.debug(f'For text chunk \n\n{text}')
        logger.debug(f'Entities found \n\n {[ent.get("surface") for ent in entities_found]}')
        logger.debug(f'Acronyms found \n\n {[key for key, _ in acronyms_found.items()]}')

    return chunks


def enrich_document_chunks(doc_id: str, acronyms: Dict[str, str], entities: List[Dict]) -> None:
    storage_context = load_index().storage_context
    docstore = storage_context.docstore

    info = docstore.get_ref_doc_info(doc_id)
    if not info:
        logger.warning(f"No nodes found for doc_id={doc_id}")
        return

    # enrich metadata in docstore
    node_ids = info.node_ids
    nodes = docstore.get_nodes(node_ids)
    updated_nodes = []

    for node in nodes:
        text = getattr(node, "text", "")

        acronyms_found = [
            {"short": acr, "long": expansion}
            for acr, expansion in acronyms.items()
            if acr in text or expansion in text
        ]
        entities_found = [
            ent for ent in entities
            if ent.get("surface") and ent["surface"].lower() in text.lower()
        ]

        node.metadata["acronyms"] = acronyms_found
        node.metadata["entities"] = entities_found
        updated_nodes.append(node)

        logger.debug(f'For text chunk \n\n{text[:200]}...')
        logger.debug(f'Entities found: {[ent.get("surface") for ent in entities_found]}')
        logger.debug(f'Acronyms found: {[a.get("short") for a in acronyms_found]}')

    docstore.add_documents(updated_nodes)

    storage_context.persist()
    logger.info(f"Enriched and persisted {len(updated_nodes)} chunks for document {doc_id}.")



def main():
    """
    Run with `python -m src.storage --file <file_id> --reset` 
    Expects file in ./output/
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Add a MinerU JSON file to the docstore + Chroma vector store."
    )
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the MinerU JSON file"
    )
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset and rebuild the index from scratch before adding the file"
    )

    args = parser.parse_args()

    if args.reset:
        reset_storage()
    
    file_id = args.file
    file_path = f'output/{file_id}/auto/{file_id}_content_list.json'

    add_file(file_path, kg_id=file_id)

if __name__ == "__main__":
    # main()
    import json 

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    with open('logs/pipeline_results.json', 'r') as f:
        results = json.load(f)

    doc_id = results['doc_id']
    entities = results['entities']
    acronyms = results['acronyms']
    enrich_document_chunks(doc_id, acronyms, entities)