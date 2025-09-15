from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
load_dotenv(dotenv_path="secrets/.env")

file_path = Path('output/test/auto/test.md')

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete vector store if it exists
chroma_client.delete_collection("test")

# Create collection
collection = chroma_client.create_collection("test")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

with open(file_path, "r") as f:
    md_string = f.read()

doc = Document(
    text=md_string,
    metadata={"source": str(file_path)})

parser = MarkdownElementNodeParser(
    include_metadata=True, 
    include_prev_next_rel=True
)

nodes = parser.get_nodes_from_documents([doc])
print(f"Parsed {len(nodes)} nodes")

index = VectorStoreIndex(nodes, storage_context=storage_context)
index.storage_context.persist(persist_dir="./storage")

docstore_count = len(storage_context.docstore.docs)
print(f"Docstore has {docstore_count} nodes")

assert len(nodes) == docstore_count, \
    f"Mismatch: parser created {len(nodes)} but docstore has {docstore_count}"
