import shutil
import logging
from pathlib import Path
from typing import Dict, List
import json
import time

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from rdflib import Graph, Namespace, RDF, Literal

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

class LlamaStorage:
    _instance = None
    _initialized = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaStorage, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._storage_context = None
            self._index = None

            self._init_storage()
            self._initialized = True

    def _init_storage(self) -> StorageContext:
        """Initialize a storage context with docstore + Chroma vector store."""
        logger.info("Initializing LlamaIndex storage context...")
        
        # Set up the Vector Store (ChromaDB)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # Load other stores from disk if they exist
        try:
            self._storage_context = StorageContext.from_defaults(
                persist_dir=str(STORAGE_DIR),
                vector_store=vector_store 
            )
            logger.info("Loaded existing storage context from disk.")
        except FileNotFoundError:
            logger.info("No existing storage found; creating new storage context.")
            self._storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

        # Initialize the Index
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=self._storage_context
        )
        logger.info("LlamaIndex is ready.")

    @property
    def index(self):
        if self._index is None:
            raise ValueError("Index not initialized.")
        return self._index

    @property
    def context(self):
        if self._storage_context is None:
            raise ValueError("Storage context not initialized.")
        return self._storage_context
        
    def persist(self):
        """
        Saves the Docstore and IndexStore to disk (JSON files).
        ChromaDB saves automatically, but LlamaIndex metadata needs this.
        """
        self.context.persist(persist_dir=str(STORAGE_DIR))
        logger.info("Persisted storage to disk.")
        

def _process_file(
        file_path: Path, 
        parser: CustomParser, 
        kg_id: str
    ) -> str:
    """Parse file into a Document + Nodes, index them, and add to storage."""
    storage = LlamaStorage()

    raw_text = file_path.read_text(encoding="utf-8")

    # Create Document
    doc = Document(
        text=raw_text, 
        metadata={"source": str(file_path)}, 
        doc_id=kg_id
    )

    # Parse into TextNodes
    nodes = parser.get_nodes_from_documents([doc])

    for n in nodes:
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc.doc_id)

    # Store nodes documents and nodes in doc store
    storage.context.docstore.add_documents([doc])
    storage.index.insert_nodes(nodes)

    return doc.doc_id


def add_file(
        file_path: str, 
        kg_id: str
    ) -> str:
    """
    Add a file to the docstore + Chroma vector store.
    """
    storage = LlamaStorage()

    parser = CustomParser(include_metadata=True, include_prev_next_rel=True)

    doc_id = _process_file(Path(file_path), parser, kg_id)

    storage.persist()

    logger.info(f"Added document {doc_id} from {file_path}.")
    
    return doc_id


# def load_existing_index() -> VectorStoreIndex:
#     chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
#     collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    
#     vector_store = ChromaVectorStore(chroma_collection=collection)
#     storage_context = StorageContext.from_defaults(
#         persist_dir=str(STORAGE_DIR),
#         vector_store=vector_store
#     )
    
#     index = load_index_from_storage(storage_context)
    
#     return index


def annotate_chunk(
        chunk: BaseNode, 
        acronyms: Dict[str, str], 
        entities: List[Dict]
    ) -> BaseNode:
    """
    Annotate each chunk with acronyms and entities that appear in its text.

    Parameters
    ----------
    chunk : BaseNode
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
    BaseNode
        Same chunk, but with added keys:
        - "acronyms": list of acronyms present
        - "entities": list of entity dicts present
    """
    text = chunk.get_content()

    acronyms_found = [
        {"short": acr, "long": expansion}
        for acr, expansion in acronyms.items()
        if acr in text or expansion in text
    ]

    entities_found = [
        ent for ent in entities
        if ent.get("surface") and ent["surface"].lower() in text.lower()
    ]

    chunk.metadata["acronyms"] = json.dumps(acronyms_found)
    chunk.metadata["entities"] = json.dumps(entities_found)  

    logger.debug(f'For text chunk \n\n{text[:200]}...')
    logger.debug(f'Entities found: {[ent.get("surface") for ent in entities_found]}')
    logger.debug(f'Acronyms found: {[a.get("short") for a in acronyms_found]}')

    return chunk


def enrich_document_chunks(
        doc_id: str, 
        acronyms: Dict[str, str], 
        entities: List[Dict]
    ) -> None:
    storage = LlamaStorage()

    ref_doc_info = storage.context.docstore.get_ref_doc_info(doc_id)
    
    if not ref_doc_info:
        logger.warning(f"No nodes found for doc_id={doc_id}")
        return

    # Enrich metadata in docstore
    node_ids = ref_doc_info.node_ids
    nodes = storage.context.docstore.get_nodes(node_ids)
    
    updated_nodes = []

    for node in nodes:
        updated_node = annotate_chunk(node, acronyms, entities)
        updated_nodes.append(updated_node)

    storage.index.delete_ref_doc(doc_id, delete_from_docstore=True)
    storage.index.insert_nodes(updated_nodes)

    storage.persist()

    logger.info(f"Enriched and persisted {len(updated_nodes)} chunks for document {doc_id}.")


def add_communities_from_graph(kg):
    storage = LlamaStorage()

    if not kg.loaded:
        logger.warning("KnowledgeGraph not loaded; cannot add communities.")
        return

    storage_context = storage.context
    docstore = storage_context.docstore
    graph = kg.g
    schema = kg.schema

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    existing_ids = set(collection.get()["ids"])

    added = 0
    new_docs = []

    for community_uri in graph.subjects(RDF.type, schema.Community):
        abstract = next(graph.objects(community_uri, schema.abstract), None)
        name = next(graph.objects(community_uri, schema.name), None)
        identifier = next(graph.objects(community_uri, schema.identifier), None)

        if not abstract:
            continue

        community_id = str(identifier or community_uri).split("/")[-1]
        doc_id = f"community-{community_id}"

        if doc_id in existing_ids:
            logger.debug(f"Skipping already indexed community {doc_id}")
            continue

        doc = Document(
            text=str(abstract),
            doc_id=doc_id,
            metadata={
                "uri": str(community_uri),
                "type": "community_summary",
                "name": str(name or ""),
                "identifier": str(identifier or ""),
            },
        )

        new_docs.append(doc)
        added += 1

    if new_docs:
        embed_model = Settings.embed_model

        for doc in new_docs:
            try:
                doc.embedding = embed_model.get_text_embedding(doc.text)
            except Exception as e:
                logger.warning(f"Failed to embed doc {doc.doc_id}: {e}")
                continue

        docstore.add_documents(new_docs)
        storage_context.vector_store.add(nodes=new_docs)

        storage_context.persist(persist_dir=str(STORAGE_DIR))
        logger.info(f"Added {added} new community summaries to existing vector store.")
    else:
        logger.info("No new community summaries to add.")



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

    storage = LlamaStorage()
    if args.reset:
        storage.reset_storage()
    
    file_id = args.file
    file_path = f'output/{file_id}/auto/{file_id}_content_list.json'

    add_file(file_path, kg_id=file_id)


if __name__ == "__main__":
    # main()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    storage = LlamaStorage()
    storage.reset()

    doc_id = '10170637'
    json_file_path = Path(f'output/{doc_id}/auto/{doc_id}_content_list.json')
    add_file(json_file_path, kg_id=doc_id)

    # from src.graph import KnowledgeGraph

    # kg = KnowledgeGraph.load_or_build('world-bank-kg.ttl', rebuild=False)
    
    # acronyms = {
    #     "SEMARNAT": "Secretaria de Medio Ambiente y Recursos Naturales",
    #     "UNBIS": "United Nations Bibliographic Information System"
    # }

    # entities = [
    #     {"surface": "World Bank", "label": "ORG", "qid": "Q123", "safe_id": "World_Bank"},
    #     {"surface": "SEMARNAT", "label": "ORG", "qid": "Q999", "safe_id": "SEMARNAT"},
    # ]
    
    # enrich_document_chunks(
    #     doc_id="10170637", 
    #     acronyms=acronyms, 
    #     entities=entities
    # )