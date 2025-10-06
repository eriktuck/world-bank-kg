import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

import src.storage as storage


# ------------------------------------------------------
# Fixtures
# ------------------------------------------------------
@pytest.fixture
def fake_doc_nodes():
    node1 = MagicMock()
    node1.text = "World Bank supports SEMARNAT in Mexico."
    node1.metadata = {}
    node2 = MagicMock()
    node2.text = "The United Nations oversees STEP and IPF projects."
    node2.metadata = {}
    return [node1, node2]


@pytest.fixture
def acronyms():
    return {"IPF": "Investment Project Financing", "STEP": "Systematic Tracking and Exchanges in Procurement"}


@pytest.fixture
def entities():
    return [
        {"surface": "World Bank", "label": "ORG", "qid": "Q123", "safe_id": "World_Bank"},
        {"surface": "SEMARNAT", "label": "ORG", "qid": "Q999", "safe_id": "SEMARNAT"},
    ]


# ------------------------------------------------------
# reset_storage
# ------------------------------------------------------
@patch("src.storage.shutil.rmtree")
@patch("src.storage.STORAGE_DIR", Path("/tmp/storage"))
@patch("src.storage.CHROMA_DIR", Path("/tmp/chroma"))
def test_reset_storage_removes_dirs(mock_rmtree):
    """Ensure both storage directories are deleted when present."""
    (Path("/tmp/storage")).mkdir(parents=True, exist_ok=True)
    (Path("/tmp/chroma")).mkdir(parents=True, exist_ok=True)

    storage.reset_storage()
    # We expect two calls: one for storage, one for chroma
    assert mock_rmtree.call_count >= 2


# ------------------------------------------------------
# _init_storage
# ------------------------------------------------------
@patch("src.storage.chromadb.PersistentClient")
@patch("src.storage.SimpleDocumentStore")
@patch("src.storage.ChromaVectorStore")
@patch("src.storage.StorageContext.from_defaults")
def test_init_storage_creates_vector_store(mock_from_defaults, mock_chroma_store, mock_docstore, mock_client):
    mock_client.return_value.get_or_create_collection.return_value = MagicMock()
    storage._init_storage()
    mock_from_defaults.assert_called_once()
    mock_chroma_store.assert_called_once()
    mock_docstore.assert_called_once()


# ------------------------------------------------------
# _process_file
# ------------------------------------------------------
@patch("src.storage.VectorStoreIndex")
@patch("pathlib.Path.read_text", return_value="Some example text.")
def test_process_file_adds_doc_and_nodes(mock_read, mock_index):
    """Ensure _process_file adds both Document and Nodes to the docstore."""
    mock_storage_context = MagicMock()
    mock_parser = MagicMock()
    fake_node = MagicMock()
    fake_node.relationships = {}
    mock_parser.get_nodes_from_documents.return_value = [fake_node]

    doc_id = storage._process_file(
        Path("fake.txt"), mock_storage_context, mock_parser, "KG001"
    )

    assert doc_id == "KG001"
    mock_parser.get_nodes_from_documents.assert_called_once()
    mock_storage_context.docstore.add_documents.assert_any_call([fake_node])
    mock_index.assert_called_once()


# ------------------------------------------------------
# add_file
# ------------------------------------------------------
@patch("src.storage._process_file", return_value="DOC123")
@patch("src.storage._init_storage")
@patch("src.storage.CustomParser")
def test_add_file_runs_pipeline(mock_parser_cls, mock_init, mock_process):
    mock_storage_context = MagicMock()
    mock_init.return_value = mock_storage_context

    result = storage.add_file("fake.txt", "DOC123")
    assert result == "DOC123"
    mock_process.assert_called_once()
    mock_storage_context.persist.assert_called_once()


# ------------------------------------------------------
# load_index
# ------------------------------------------------------
@patch("src.storage.chromadb.PersistentClient")
@patch("src.storage.ChromaVectorStore")
@patch("src.storage.StorageContext.from_defaults")
@patch("src.storage.load_index_from_storage")
def test_load_index_retrieves_vector_index(mock_load, mock_context, mock_chroma, mock_client):
    """Ensure load_index builds a proper index from persisted files."""
    mock_client.return_value.get_or_create_collection.return_value = MagicMock()
    storage.load_index()
    mock_client.assert_called_once()
    mock_chroma.assert_called_once()
    mock_load.assert_called_once()


# ------------------------------------------------------
# load_document
# ------------------------------------------------------
@patch("src.storage.StorageContext.from_defaults")
def test_load_document_returns_doc(mock_context):
    mock_docstore = MagicMock()
    mock_context.return_value.docstore = mock_docstore
    mock_docstore.get_document.return_value = {"id": "DOC1"}
    result = storage.load_document("storage", "DOC1")
    assert result == {"id": "DOC1"}
    mock_docstore.get_document.assert_called_once_with("DOC1")


# ------------------------------------------------------
# enrich_chunks_with_annotations
# ------------------------------------------------------
def test_enrich_chunks_with_annotations_adds_matches(acronyms, entities):
    chunks = [
        {"text": "The World Bank funds SEMARNAT programs and STEP initiatives."}
    ]
    enriched = storage.enrich_chunks_with_annotations(chunks, acronyms, entities)

    assert "acronyms_found" in enriched[0]
    assert "entities_found" in enriched[0]
    assert any(ent["surface"] == "World Bank" for ent in enriched[0]["entities_found"])
    assert "STEP" in enriched[0]["acronyms_found"]


def test_enrich_chunks_with_annotations_empty(acronyms, entities):
    chunks = [{"text": "Completely unrelated text"}]
    enriched = storage.enrich_chunks_with_annotations(chunks, acronyms, entities)
    assert enriched[0]["acronyms_found"] == {}
    assert enriched[0]["entities_found"] == []


# ------------------------------------------------------
# enrich_document_chunks
# ------------------------------------------------------
@patch("src.storage.load_index")
def test_enrich_document_chunks_updates_nodes(mock_load_index, acronyms, entities):
    """Should enrich each node's metadata with detected acronyms + entities."""
    node1 = MagicMock()
    node1.text = "The World Bank works with SEMARNAT and STEP."
    node1.metadata = {}
    node2 = MagicMock()
    node2.text = "IPF and STEP are World Bank programs."
    node2.metadata = {}

    fake_doc_nodes = [node1, node2]

    mock_docstore = MagicMock()
    mock_info = MagicMock()
    mock_info.node_ids = [1, 2]

    mock_docstore.get_ref_doc_info.return_value = mock_info
    mock_docstore.get_nodes.return_value = fake_doc_nodes

    mock_storage_context = MagicMock(docstore=mock_docstore)
    mock_index = MagicMock(storage_context=mock_storage_context)
    mock_load_index.return_value = mock_index

    storage.enrich_document_chunks("DOC100", acronyms, entities)

    # Check metadata enrichment
    for node in fake_doc_nodes:
        assert "acronyms" in node.metadata
        assert "entities" in node.metadata
        assert all("surface" in e for e in node.metadata["entities"])

    mock_docstore.get_ref_doc_info.assert_called_once_with("DOC100")
    mock_storage_context.persist.assert_called_once()


@patch("src.storage.load_index")
def test_enrich_document_chunks_warns_if_no_info(mock_load_index, caplog):
    mock_index = MagicMock()
    mock_index.storage_context.docstore.get_ref_doc_info.return_value = None
    mock_load_index.return_value = mock_index

    storage.enrich_document_chunks("BAD_DOC", {}, [])
    assert "No nodes found" in caplog.text
