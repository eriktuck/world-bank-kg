import shutil
import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.parser import CustomParser

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path="secrets/.env")

STORAGE_DIR = Path("./storage")
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "documents"


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


def _process_file(file_path: Path, storage_context: StorageContext, parser: CustomParser) -> int:
    """Parse file into a Document + Nodes, index them, and add to storage."""
    raw_text = Path(file_path).read_text()

    # Create Document
    doc = Document(text=raw_text, metadata={"source": str(file_path)})
    storage_context.docstore.add_documents([doc])

    # Parse into TextNodes
    nodes = parser.get_nodes_from_documents([doc])

    # Insert into VectorStore
    VectorStoreIndex(nodes, storage_context=storage_context)

    return len(nodes)


def add_file(file_path: str, reset: bool = False):
    """
    Add a file to the docstore + Chroma vector store.
    If reset=True, delete existing stores and rebuild from scratch.
    """
    if reset:
        reset_storage()

    storage_context = _init_storage()
    parser = CustomParser(include_metadata=True, include_prev_next_rel=True)

    n_nodes = _process_file(Path(file_path), storage_context, parser)

    storage_context.persist(persist_dir=str(STORAGE_DIR))

    action = "Rebuilt index" if reset else "Added"
    logger.info(f"{action} with {n_nodes} nodes from {file_path}.")


def main():
    """Run with `python -m src.storage --file output/test/auto/test_content_list.json --reset` """
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
    add_file(args.file, reset=args.reset)


if __name__ == "__main__":
    main()