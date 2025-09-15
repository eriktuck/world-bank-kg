from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
load_dotenv(dotenv_path="secrets/.env")

QUERY = "Why is the World Bank involved in this project?"

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("test")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=3,      # adjust how many context nodes to retrieve
    response_mode="tree_summarize",  # or "compact", "accumulate", etc.
    include_text=True        # ensures node text is included in sources
)

response = query_engine.query(QUERY)

# --- Results ---
print("ANSWER:\n", response.response)

print("\n--- CONTEXT NODES USED ---")
for node in response.source_nodes:
    print(f"[Score: {node.score:.2f}] {node.node.text[:200]}...\n")