import json
import logging
from pathlib import Path
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from src.graph import KnowledgeGraph
from src.storage import CHROMA_DIR, STORAGE_DIR, COLLECTION_NAME, Settings, reset_storage, add_file, add_communities_from_graph


logger = logging.getLogger(__name__)

def rebuild_vector_index(
    doc_id: str = '10170637',
    ttl_path: str = "world-bank-kg.ttl"
):
    """Rebuild the Chroma vector index from cached outputs (no re-parsing needed)."""

    reset_storage()

    json_file_path = Path(f'output/{doc_id}/auto/{doc_id}_content_list.json')
    add_file(json_file_path, kg_id=doc_id)

    kg = KnowledgeGraph.load_or_build(ttl_path, rebuild=False)
    add_communities_from_graph(kg)
    
    logger.info("✅ Rebuilt vector index successfully!")

if __name__ == '__main__':
    rebuild_vector_index()