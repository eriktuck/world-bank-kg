import shutil
import logging
from pathlib import Path
from typing import Dict, List
import json
import time
import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Document, 
    StorageContext, 
    VectorStoreIndex, 
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from rdflib import Graph, Namespace, RDF, Literal

from src.parser import CustomParser
from src import config

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="secrets/.env")

# Sourced from src.config so every storage path/name has a single source of truth.
STORAGE_DIR = config.LINDEX_STORAGE_PATH
CHROMA_DIR = config.CHROMA_PATH
COLLECTION_NAME = config.COLLECTION_NAME


from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

class LlamaStorage:
    def __init__(self):
        """
        Initializes the database connection.
        If data exists, it loads it. If not, it sets up a fresh index.
        """
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.chroma_collection = self.chroma_client.get_or_create_collection(config.COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index from storage, or create a new one if not found."""
        docstore_path = os.path.join(config.LINDEX_STORAGE_PATH, "docstore.json")
        
        if os.path.exists(docstore_path):
            logger.info("Loading existing storage context from disk...")

            storage_context = StorageContext.from_defaults(
                persist_dir=config.LINDEX_STORAGE_PATH,
                vector_store=self.vector_store 
            )
            index = load_index_from_storage(storage_context)

        else:
            logger.info("Creating new empty index...")
            
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            index = VectorStoreIndex.from_vector_store(
                self.vector_store, 
                storage_context=storage_context,
                store_nodes_override=True # Important: keeps text in JSON, vectors in Chroma
            )
        
        return index

    @property
    def context(self):
        return self.index.storage_context

    def persist(self):
        """
        Saves the Docstore and IndexStore to disk (JSON files).
        ChromaDB saves automatically, but LlamaIndex metadata needs this.
        """
        target_dir = config.LINDEX_STORAGE_PATH
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        self.context.persist(persist_dir=str(target_dir))
        logger.info(f"Persisted storage to {target_dir}.")
        
    def _process_and_insert(
        self,
        file_path: Path, 
        parser: CustomParser, 
        kg_id: str
    ) -> str:
        """Parse file into a Document + Nodes, index them, and add to storage."""
        raw_text = file_path.read_text(encoding="utf-8")

        # Create Document
        doc = Document(
            text=raw_text, 
            metadata={"source": str(file_path)}, 
            doc_id=kg_id
        )

        # Parse into TextNodes
        nodes = parser.get_nodes_from_documents([doc])

        # Store nodes documents and nodes in doc store
        self.context.docstore.add_documents([doc])
        self.index.insert_nodes(nodes)

        return doc.doc_id

    def add_file(
        self,
        file_path: str, 
        kg_id: str
    ) -> str:
        """
        Add a file to the docstore + Chroma vector store.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        parser = CustomParser(include_metadata=True, include_prev_next_rel=True)

        logger.info(f"Processing {file_path}...")

        self._process_and_insert(path_obj, parser, kg_id)
        self.persist()

        logger.info(f"Successfully added document {kg_id}.")
        
        return kg_id

    def clear(self):
        """Nuke it. Useful for resetting."""
        print("Clearing all data...")
        
        # Delete Chroma collection
        self.chroma_client.delete_collection(config.COLLECTION_NAME)
        
        # Delete local files
        if os.path.exists(config.LINDEX_STORAGE_PATH):
            shutil.rmtree(config.LINDEX_STORAGE_PATH)
        
        print("Storage cleared.")

    


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

    collection = storage.chroma_collection
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

        storage.persist()
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
        storage.clear()
        storage = LlamaStorage()

    file_id = args.file
    file_path = f'output/{file_id}/auto/{file_id}_content_list.json'

    storage.add_file(file_path, kg_id=file_id)


if __name__ == "__main__":
    # main()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    storage = LlamaStorage()

    doc_id = '10170637'
    json_file_path = Path(f'output/{doc_id}/auto/{doc_id}_content_list.json')
    storage.add_file(json_file_path, kg_id=doc_id)

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