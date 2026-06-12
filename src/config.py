import os

# Where to save the database
PERSIST_DIR = "./data_storage"
CHROMA_PATH = os.path.join(PERSIST_DIR, "chroma_db")
LINDEX_STORAGE_PATH = os.path.join(PERSIST_DIR, "docstore")
COLLECTION_NAME = "documents"

# Model Settings (Optional, but good practice to be explicit)
EMBEDDING_MODEL = "local:BAAI/bge-small-en-v1.5" # or "openai"
LLM_MODEL = "gpt-3.5-turbo"