import json
from pathlib import Path
import pytest

from llama_index.core import Document
from llama_index.core.schema import NodeRelationship
from models.custom_parser import CustomParser

@pytest.fixture
def sample_document():
    """Load the structured sample fixture and wrap in a LLamaIndex Document"""
    fixture_path = Path(__file__).parent / "fixtures" / "structured_sample.json"
    with open(fixture_path, 'r') as f:
        struct_out = f.read()

    return Document(
        text = struct_out,
        metadata={"source": str(fixture_path)}
    )

def test_cusomtParser_creates_nodes(sample_document):
    parser = CustomParser(
        include_metadata=True,
        include_prev_next_rel=True
    )

    nodes = parser.get_nodes_from_documents([sample_document])

    # Basic checks
    assert isinstance(nodes, list)
    assert len(nodes) == 9

    # ---- Text nodes ----
    text_nodes = [n for n in nodes if "Wind Umbrella" in n.text or "Key Safeguards" in n.text]
    assert any("Wind Umbrella" in tn.text for tn in text_nodes)
    assert any("Key Safeguards" in tn.text for tn in text_nodes)
    assert all("page_idx" in tn.metadata for tn in text_nodes)

    # ---- Table node ----
    table_nodes = [n for n in nodes if "<table>" in n.text]
    assert len(table_nodes) == 1
    table_node = table_nodes[0]
    assert "Safeguard Policies Triggered by the Project" in table_node.text
    assert table_node.metadata["page_idx"] == 22

    # ---- Image node ----
    image_nodes = [n for n in nodes if n.text.endswith(".jpg")]
    assert len(image_nodes) == 1
    image_node = image_nodes[0]
    assert "Annex 15" in " ".join(image_node.metadata.get("image_caption", []))
    assert image_node.metadata["page_idx"] == 59



def test_custom_parser_includes_prev_next(sample_document):
    parser = CustomParser(include_metadata=True, include_prev_next_rel=True)
    nodes = parser.get_nodes_from_documents([sample_document])

    assert nodes[0].relationships.get(NodeRelationship.NEXT) is not None
    assert nodes[-1].relationships.get(NodeRelationship.PREVIOUS) is not None

    # First node has no previous
    assert NodeRelationship.PREVIOUS not in nodes[0].relationships
    # Last node has no next
    assert NodeRelationship.NEXT not in nodes[-1].relationships

