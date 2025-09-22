import json
from pathlib import Path
import pytest
import os

from llama_index.core import Document
from llama_index.core.schema import NodeRelationship
from parser import CustomParser

import json

def ensure_list(val):
    """Normalize metadata fields that may be JSON-encoded strings or already lists."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return [val]
    return [val]


@pytest.fixture
def sample_document():
    """Load the structured sample fixture and wrap in a LLamaIndex Document"""
    fixture_path = Path(__file__).parent / "fixtures" / "structured_sample.json"
    with open(fixture_path, 'r') as f:
        struct_out = f.read()

    return Document(
        text=struct_out,
        metadata={"source": str(fixture_path)}
    )

def normalize_pages(pages):
    """Normalize page_idx to a list of ints"""
    if isinstance(pages, list):
        return pages
    if isinstance(pages, str):
        return [int(p.strip()) for p in pages.split(",") if p.strip().isdigit()]
    if isinstance(pages, int):
        return [pages]
    return []

def test_customParser_creates_nodes(sample_document):
    parser = CustomParser(
        include_metadata=True,
        include_prev_next_rel=True
    )
    nodes = parser.get_nodes_from_documents([sample_document])

    assert isinstance(nodes, list)

    # ---- Text nodes ----
    text_nodes = [
        n for n in nodes
        if "Wind Umbrella" in n.text or "Key Safeguards" in str(n.metadata.get("headers", ""))
    ]
    assert any("Wind Umbrella" in tn.text for tn in text_nodes)
    assert any(
        str(n.metadata.get("headers", "")).startswith("Key Safeguards")
        for n in nodes
    )

    # ---- Table node ----
    table_nodes = [n for n in nodes if "<table>" in n.text]
    assert len(table_nodes) == 1
    table_node = table_nodes[0]
    assert "Safeguard Policies Triggered by the Project" in table_node.text
    assert int(table_node.metadata["page_idx"]) == 22
    assert str(table_node.metadata.get("headers", "")).startswith("6. Safeguard policies")

    # ---- Image node ----
    image_nodes = [n for n in nodes if n.text.endswith(".jpg")]
    assert len(image_nodes) == 1
    image_node = image_nodes[0]
    captions = ensure_list(image_node.metadata.get("image_caption", []))
    assert any("Annex 15" in cap for cap in captions)
    assert int(image_node.metadata["page_idx"]) == 59
    assert str(image_node.metadata.get("headers", "")).startswith("6. Safeguard policies")



def test_custom_parser_prev_next_after_chunking():
    long_text = "Sentence one is long. Sentence two is long. Sentence three is long."
    sample_json = [{"type": "text", "text": long_text, "page_idx": 1}]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=True)
    nodes = parser.get_nodes_from_documents([doc], max_chars=40)

    assert len(nodes) > 1
    for i, node in enumerate(nodes):
        if i > 0:
            prev_info = node.relationships.get(NodeRelationship.PREVIOUS)
            assert prev_info is not None
            assert prev_info.node_id == nodes[i - 1].node_id
        if i < len(nodes) - 1:
            next_info = node.relationships.get(NodeRelationship.NEXT)
            assert next_info is not None
            assert next_info.node_id == nodes[i + 1].node_id


def test_custom_parser_attaches_headers():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "First paragraph under A.", "page_idx": 1},
        {"type": "text", "text": "Header B", "text_level": 1, "page_idx": 2},
        {"type": "text", "text": "First paragraph under B.", "page_idx": 2},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) == 2
    first, second = nodes
    assert first.metadata.get("headers") == "Header A"
    assert "First paragraph under A" in first.text
    assert second.metadata.get("headers") == "Header B"
    assert "First paragraph under B" in second.text


def test_custom_parser_concatenates_short_texts():
    sample_json = [
        {"type": "text", "text": "Sentence one.", "page_idx": 1},
        {"type": "text", "text": "Sentence two.", "page_idx": 1},
        {"type": "text", "text": "Sentence three.", "page_idx": 1},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=200)

    assert len(nodes) == 1
    combined_text = nodes[0].text
    assert "Sentence one." in combined_text
    assert "Sentence two." in combined_text
    assert "Sentence three." in combined_text


def test_custom_parser_flushes_on_headers_and_special_nodes():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "First under A.", "page_idx": 1},
        {"type": "text", "text": "Second under A.", "page_idx": 1},
        {"type": "table", "table_body": "<table><tr><td>A Table</td></tr></table>", "page_idx": 1},
        {"type": "text", "text": "Header B", "text_level": 1, "page_idx": 2},
        {"type": "text", "text": "First under B.", "page_idx": 2},
        {"type": "image", "img_path": "image.png", "page_idx": 2},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=500)

    assert len(nodes) == 4
    assert "First under A." in nodes[0].text
    assert "Second under A." in nodes[0].text
    assert nodes[0].metadata.get("headers") == "Header A"
    assert "<table>" in nodes[1].text
    assert nodes[1].metadata.get("headers") == "Header A"
    assert "First under B." in nodes[2].text
    assert nodes[2].metadata.get("headers") == "Header B"
    assert nodes[3].text == "image.png"
    assert nodes[3].metadata.get("headers") == "Header B"


def test_custom_parser_flushes_on_new_header_boundaries():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "First under A.", "page_idx": 1},
        {"type": "text", "text": "Header B", "text_level": 1, "page_idx": 2},
        {"type": "text", "text": "First under B.", "page_idx": 2},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=500)

    assert len(nodes) == 2
    assert "First under A." in nodes[0].text
    assert nodes[0].metadata.get("headers") == "Header A"
    assert "First under B." in nodes[1].text
    assert nodes[1].metadata.get("headers") == "Header B"


def test_custom_parser_splits_overly_long_text():
    long_text = "Sentence one is long. Sentence two is also long. Sentence three is very, very long."
    sample_json = [{"type": "text", "text": long_text, "page_idx": 1}]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=40)

    assert len(nodes) > 1
    combined_texts = " ".join(node.text for node in nodes)
    assert "Sentence one" in combined_texts
    assert "Sentence two" in combined_texts
    assert "Sentence three" in combined_texts
    assert all(len(node.text) <= 40 for node in nodes)


def test_custom_parser_keeps_tables_and_images_atomic():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "Intro text under A.", "page_idx": 1},
        {"type": "table", "table_body": "<table><tr><td>Some Table</td></tr></table>", "page_idx": 1},
        {"type": "text", "text": "More text under A.", "page_idx": 1},
        {"type": "image", "img_path": "figure.png", "image_caption": ["A figure"], "page_idx": 1},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=500)

    assert len(nodes) == 4
    assert "Intro text under A." in nodes[0].text
    assert nodes[0].metadata.get("headers") == "Header A"
    assert "<table>" in nodes[1].text
    assert nodes[1].metadata.get("headers") == "Header A"
    assert "More text under A." in nodes[2].text
    assert nodes[2].metadata.get("headers") == "Header A"
    assert nodes[3].text == "figure.png"

    captions = ensure_list(nodes[3].metadata.get("image_caption", []))
    assert any("A figure" in cap for cap in captions)



def test_custom_parser_skips_empty_text_nodes():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "", "page_idx": 1},
        {"type": "text", "text": "Non-empty paragraph.", "page_idx": 1},
        {"type": "text", "text": "   ", "page_idx": 1},
        {"type": "text", "text": "Another paragraph.", "page_idx": 1},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=200)

    assert len(nodes) == 1
    text = nodes[0].text
    assert "Non-empty paragraph." in text
    assert "Another paragraph." in text
    assert text.strip() != ""
    assert "\n\n\n\n" not in text
    assert nodes[0].metadata.get("headers") == "Header A"


def test_custom_parser_propagates_metadata_across_nodes():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "First paragraph.", "page_idx": 1},
        {"type": "text", "text": "Second paragraph.", "page_idx": 2},
        {"type": "table", "table_body": "<table><tr><td>Some Table</td></tr></table>", "page_idx": 3, "table_caption": ["A Table"]},
        {"type": "image", "img_path": "figure.png", "image_caption": ["A Figure"], "page_idx": 4},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=500)

    assert len(nodes) == 3
    text_node, table_node, image_node = nodes

    pages = normalize_pages(text_node.metadata.get("page_idx"))
    assert 1 in pages and 2 in pages
    assert "First paragraph." in text_node.text
    assert "Second paragraph." in text_node.text

    assert int(table_node.metadata["page_idx"]) == 3
    captions = ensure_list(table_node.metadata.get("table_caption", []))
    assert any("A Table" in cap for cap in captions)
    assert "<table>" in table_node.text

    assert int(image_node.metadata["page_idx"]) == 4
    captions = ensure_list(image_node.metadata.get("image_caption", []))
    assert any("A Figure" in cap for cap in captions)
    assert image_node.text == "figure.png"



def test_custom_parser_never_merges_across_headers():
    sample_json = [
        {"type": "text", "text": "H1 A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "Para A1.", "page_idx": 1},
        {"type": "text", "text": "Para A2.", "page_idx": 1},
        {"type": "text", "text": "H2 B", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "Para B1.", "page_idx": 1},
        {"type": "text", "text": "Para B2.", "page_idx": 1},
        {"type": "text", "text": "H2 C", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "Para C1.", "page_idx": 1},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=10_000)

    assert len(nodes) == 3
    assert "Para A1." in nodes[0].text and "Para A2." in nodes[0].text
    assert nodes[0].metadata.get("headers") == "H1 A"
    assert "Para B1." in nodes[1].text and "Para B2." in nodes[1].text
    assert "H1 A" in str(nodes[1].metadata.get("headers"))
    assert "H2 B" in str(nodes[1].metadata.get("headers"))
    assert "Para C1." in nodes[2].text
    assert "H1 A" in str(nodes[2].metadata.get("headers"))
    assert "H2 C" in str(nodes[2].metadata.get("headers"))


def test_custom_parser_golden_output(sample_document, update_golden):
    parser = CustomParser(
        include_metadata=True,
        include_prev_next_rel=True,
        chunk_overlap=20,
    )
    nodes = parser.get_nodes_from_documents([sample_document], max_chars=200)

    def normalize(node):
        metadata = dict(node.metadata)
        metadata["page_idx"] = normalize_pages(metadata.get("page_idx"))
        for key in ["image_caption", "image_footnote", "table_caption"]:
            if key in metadata:
                metadata[key] = ensure_list(metadata[key])
        rels = [k.name for k in node.relationships.keys()]
        return {
            "text": node.text,
            "metadata": metadata,
            "relationships": sorted(rels),
        }

    actual = [normalize(n) for n in nodes]
    golden_file = Path(__file__).parent / "fixtures" / "golden_output.json"

    if update_golden or not golden_file.exists():
        with open(golden_file, "w") as f:
            json.dump(actual, f, indent=2)
        pytest.skip("Golden file updated, re-run test without --update-golden")
    else:
        with open(golden_file) as f:
            expected = json.load(f)
        assert actual == expected



@pytest.mark.skipif(
    not os.getenv("RUN_DEBUG", False),
    reason="Set RUN_DEBUG=1 to enable"
)
def test_debug_parser_output():
    fixture_path = Path(__file__).parent / "fixtures" / "structured_sample.json"
    with open(fixture_path, "r") as f:
        struct_out = f.read()
    doc = Document(text=struct_out, metadata={"source": str(fixture_path)})
    parser = CustomParser(include_metadata=True, include_prev_next_rel=True)
    nodes = parser.get_nodes_from_documents([doc])

    print("\n--- DEBUG: CustomParser Output ---")
    for i, node in enumerate(nodes):
        print(f"\nNode {i}:")
        print(f"  Text (truncated): {node.text[:80]!r}")
        print(f"  Headers: {node.metadata.get('headers')}")
        print(f"  Metadata keys: {list(node.metadata.keys())}")
        print(f"  Relationships: {list(node.relationships.keys())}")
    print("\n--- END DEBUG ---")
