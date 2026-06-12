import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

# Import your class from src.storage
from src.storage import LlamaStorage

# --- Mocks ---

class MockParser:
    """
    A simple mock that acts like your CustomParser.
    It forces the text to be accepted as a node, ignoring validation rules.
    """
    def __init__(self, include_metadata=True, include_prev_next_rel=True):
        pass

    def get_nodes_from_documents(self, documents):
        nodes = []
        for doc in documents:
            # Create a simple node for every doc, bypassing "ill-formed text" checks
            node = TextNode(text=doc.text, metadata=doc.metadata)
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc.doc_id)
            nodes.append(node)
        return nodes

# --- Fixtures ---

@pytest.fixture
def mock_storage_env(tmp_path):
    """
    Sets up temporary directories and patches src.config constants.
    """
    chroma_dir = tmp_path / "chroma_db"
    storage_dir = tmp_path / "storage"
    
    # Patch the config variables so they point to tmp_path
    with patch("src.config.CHROMA_PATH", str(chroma_dir)), \
         patch("src.config.LINDEX_STORAGE_PATH", str(storage_dir)), \
         patch("src.config.COLLECTION_NAME", "test_collection"):
         
        yield {
            "chroma": chroma_dir,
            "storage": storage_dir
        }

@pytest.fixture
def mock_custom_parser():
    """
    Patches the CustomParser class used inside src.storage.
    CRITICAL: This path must match where CustomParser is IMPORTED in src/storage.py
    """
    with patch("src.storage.CustomParser", side_effect=MockParser):
        yield

# --- Tests ---

def test_initialization_and_persistence(mock_storage_env, mock_custom_parser):
    """
    Test 1: Lifecycle Check
    """
    # Setup a dummy file
    temp_file = Path(mock_storage_env["storage"]) / "test_doc.txt"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.write_text("This is a persistent test document.")

    # 1. First Initialization
    storage = LlamaStorage()
    assert storage.index is not None
    
    # 2. Add File
    # The mock parser will now ensure this text is NOT skipped
    storage.add_file(str(temp_file), kg_id="test_001")
    
    # 3. Persist (Explicitly calling to be safe, though add_file should do it)
    storage.persist()

    # Verify file exists on disk
    # If the parser worked, we should have data in docstore.json
    docstore_path = mock_storage_env["storage"] / "docstore.json"
    assert docstore_path.exists(), "docstore.json was not created!"

    # 4. Re-initialization (Simulate app restart)
    storage_v2 = LlamaStorage()
    
    # 5. Verify Data
    assert "test_001" in storage_v2.context.docstore.docs
    print("\nTest 1 Passed: Persistence works correctly.")


def test_add_file_and_query(mock_storage_env, mock_custom_parser):
    """
    Test 2: Functionality Check
    """
    # Setup dummy file
    content = "The secret code for the vault is 998877."
    temp_file = Path(mock_storage_env["storage"]) / "secret.txt"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.write_text(content)

    storage = LlamaStorage()
    storage.add_file(str(temp_file), kg_id="secret_doc")

    # Verify it's in the index
    assert len(storage.index.docstore.docs) > 0

    # Query
    retriever = storage.index.as_retriever()
    nodes = retriever.retrieve("What is the secret code?")
    
    # Check if we retrieved the correct node
    assert len(nodes) > 0
    assert "998877" in nodes[0].text
    print("\nTest 2 Passed: Indexing and Retrieval work.")