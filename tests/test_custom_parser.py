import json
from pathlib import Path
import pytest
import os

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

def test_customParser_creates_nodes(sample_document):
    parser = CustomParser(
        include_metadata=True,
        include_prev_next_rel=True
    )

    nodes = parser.get_nodes_from_documents([sample_document])

    # Basic checks
    assert isinstance(nodes, list)

    # ---- Text nodes ----
    text_nodes = [n for n in nodes if "Wind Umbrella" in n.text or "Key Safeguards" in n.metadata.get("headers", [[]])[-1][1]]
    assert any("Wind Umbrella" in tn.text for tn in text_nodes)
    # Check that a node has the header "Key Safeguards issues"
    assert any(
        headers and headers[0][1].startswith("Key Safeguards")
        for headers in (n.metadata.get("headers") for n in nodes)
    )
    assert all("page_idx" in tn.metadata for tn in text_nodes)

    # ---- Table node ----
    table_nodes = [n for n in nodes if "<table>" in n.text]
    assert len(table_nodes) == 1
    table_node = table_nodes[0]
    assert "Safeguard Policies Triggered by the Project" in table_node.text
    assert table_node.metadata["page_idx"] == 22
    assert table_node.metadata["headers"][0][1].startswith("6. Safeguard policies")

    # ---- Image node ----
    image_nodes = [n for n in nodes if n.text.endswith(".jpg")]
    assert len(image_nodes) == 1
    image_node = image_nodes[0]
    assert "Annex 15" in " ".join(image_node.metadata.get("image_caption", []))
    assert image_node.metadata["page_idx"] == 59
    assert image_node.metadata["headers"][0][1].startswith("6. Safeguard policies")


def test_custom_parser_prev_next_after_chunking():
    # A single long text will be split into multiple chunks
    long_text = (
        "Sentence one is long. "
        "Sentence two is long. "
        "Sentence three is long."
    )
    sample_json = [{"type": "text", "text": long_text, "page_idx": 1}]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})

    parser = CustomParser(include_metadata=True, include_prev_next_rel=True)
    nodes = parser.get_nodes_from_documents([doc], max_chars=40)

    # Expect more than one node because max_chars is small
    assert len(nodes) > 1

    # Check relationships
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

    # We expect 2 content nodes (the paragraphs), each with headers
    assert len(nodes) == 2

    first, second = nodes

    assert first.metadata["headers"] == [(1, "Header A")]
    assert "First paragraph under A" in first.text

    assert second.metadata["headers"] == [(1, "Header B")]
    assert "First paragraph under B" in second.text


def test_custom_parser_concatenates_short_texts():
    sample_json = [
        {"type": "text", "text": "Sentence one.", "page_idx": 1},
        {"type": "text", "text": "Sentence two.", "page_idx": 1},
        {"type": "text", "text": "Sentence three.", "page_idx": 1},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})

    # Use a generous max_chars so no flush happens
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=200)

    # Expect one node, concatenated
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

    # Expect: 4 nodes total:
    # - concatenated text under Header A
    # - table node
    # - text under Header B
    # - image node
    assert len(nodes) == 4

    # Check text chunks
    assert "First under A." in nodes[0].text
    assert "Second under A." in nodes[0].text
    assert nodes[0].metadata["headers"] == [(1, "Header A")]

    # Table flushes buffer
    assert "<table>" in nodes[1].text
    assert nodes[1].metadata["headers"] == [(1, "Header A")]

    # Header B flushes buffer
    assert "First under B." in nodes[2].text
    assert nodes[2].metadata["headers"] == [(1, "Header B")]

    # Image is its own node
    assert nodes[3].text == "image.png"
    assert nodes[3].metadata["headers"] == [(1, "Header B")]


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

    # Expect: 2 nodes, one per header section
    assert len(nodes) == 2

    assert "First under A." in nodes[0].text
    assert nodes[0].metadata["headers"] == [(1, "Header A")]

    assert "First under B." in nodes[1].text
    assert nodes[1].metadata["headers"] == [(1, "Header B")]


def test_custom_parser_splits_overly_long_text():
    # One artificially long paragraph (3 sentences concatenated)
    long_text = (
        "Sentence one is long. "
        "Sentence two is also long. "
        "Sentence three is very, very long."
    )

    sample_json = [
        {"type": "text", "text": long_text, "page_idx": 1}
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})

    # Set max_chars very small so we force splitting
    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=40)

    # Expect more than one node because the paragraph exceeds max_chars
    assert len(nodes) > 1

    # Sentences should be preserved in order across chunks
    combined_texts = " ".join(node.text for node in nodes)
    assert "Sentence one" in combined_texts
    assert "Sentence two" in combined_texts
    assert "Sentence three" in combined_texts

    # No single chunk should exceed the max_chars limit
    assert all(len(node.text) <= 40 for node in nodes)


def test_custom_parser_keeps_tables_and_images_atomic():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "Intro text under A.", "page_idx": 1},
        {
            "type": "table",
            "table_body": "<table><tr><td>Some Table</td></tr></table>",
            "page_idx": 1,
        },
        {"type": "text", "text": "More text under A.", "page_idx": 1},
        {"type": "image", "img_path": "figure.png", "image_caption": ["A figure"], "page_idx": 1},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})

    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=500)

    # Expect 4 nodes:
    # - concatenated intro text
    # - table node
    # - "More text under A." as a separate node (buffer flushed before table)
    # - image node
    assert len(nodes) == 4

    # First node: text
    assert "Intro text under A." in nodes[0].text
    assert nodes[0].metadata["headers"] == [(1, "Header A")]

    # Second node: table
    assert "<table>" in nodes[1].text
    assert nodes[1].metadata["headers"] == [(1, "Header A")]

    # Third node: text after the table
    assert "More text under A." in nodes[2].text
    assert nodes[2].metadata["headers"] == [(1, "Header A")]

    # Fourth node: image
    assert nodes[3].text == "figure.png"
    assert "A figure" in " ".join(nodes[3].metadata.get("image_caption", []))
    assert nodes[3].metadata["headers"] == [(1, "Header A")]


def test_custom_parser_skips_empty_text_nodes():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "", "page_idx": 1},           # should be skipped
        {"type": "text", "text": "Non-empty paragraph.", "page_idx": 1},
        {"type": "text", "text": "   ", "page_idx": 1},        # whitespace-only, skipped
        {"type": "text", "text": "Another paragraph.", "page_idx": 1},
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})

    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=200)

    # Because of concatenation, both non-empty texts become a single chunk
    assert len(nodes) == 1
    text = nodes[0].text
    assert "Non-empty paragraph." in text
    assert "Another paragraph." in text
    # Ensure no empty/whitespace-only nodes slipped through
    assert text.strip() != ""
    # Optional: make sure we didn't insert extra separators from empties
    assert "\n\n\n\n" not in text

    # Headers should still be attached
    assert nodes[0].metadata["headers"] == [(1, "Header A")]


def test_custom_parser_propagates_metadata_across_nodes():
    sample_json = [
        {"type": "text", "text": "Header A", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "First paragraph.", "page_idx": 1},
        {"type": "text", "text": "Second paragraph.", "page_idx": 2},
        {
            "type": "table",
            "table_body": "<table><tr><td>Some Table</td></tr></table>",
            "page_idx": 3,
            "table_caption": ["A Table"],
        },
        {
            "type": "image",
            "img_path": "figure.png",
            "image_caption": ["A Figure"],
            "page_idx": 4,
        },
    ]
    doc = Document(text=json.dumps(sample_json), metadata={"source": "synthetic"})

    parser = CustomParser(include_metadata=True, include_prev_next_rel=False)
    nodes = parser.get_nodes_from_documents([doc], max_chars=500)

    # Expect: 3 nodes -> 1 text chunk (pages 1+2), 1 table, 1 image
    assert len(nodes) == 3

    text_node, table_node, image_node = nodes

    # Text node should have page indices from both paragraphs
    pages = text_node.metadata.get("page_idx")
    assert isinstance(pages, list)
    assert 1 in pages and 2 in pages
    assert "First paragraph." in text_node.text
    assert "Second paragraph." in text_node.text

    # Table node metadata should include original page and caption
    assert table_node.metadata["page_idx"] == 3
    assert "A Table" in " ".join(table_node.metadata.get("table_caption", []))
    assert "<table>" in table_node.text

    # Image node metadata should include page and caption
    assert image_node.metadata["page_idx"] == 4
    assert "A Figure" in " ".join(image_node.metadata.get("image_caption", []))
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

    # Big max_chars so concatenation would happen if allowed
    nodes = parser.get_nodes_from_documents([doc], max_chars=10_000)

    # Expect 3 chunks: A*, B*, C* — not one giant merged chunk
    assert len(nodes) == 3

    # A*
    assert "Para A1." in nodes[0].text and "Para A2." in nodes[0].text
    assert nodes[0].metadata["headers"] == [(1, "H1 A")]

    # B*
    assert "Para B1." in nodes[1].text and "Para B2." in nodes[1].text
    assert nodes[1].metadata["headers"] == [(1, "H1 A"), (2, "H2 B")]

    # C*
    assert "Para C1." in nodes[2].text
    assert nodes[2].metadata["headers"] == [(1, "H1 A"), (2, "H2 C")]


def test_custom_parser_golden_output(sample_document, update_golden):
    parser = CustomParser(
        include_metadata=True,
        include_prev_next_rel=True,
        chunk_overlap=20,
    )
    nodes = parser.get_nodes_from_documents([sample_document], max_chars=200)

    def normalize(node):
        # Normalize headers to JSON-friendly lists
        headers = node.metadata.get("headers", [])
        norm_headers = [[lvl, text] for lvl, text in headers]

        # Normalize relationships: use relationship type names only
        rels = [k.name for k in node.relationships.keys()]

        # Copy metadata and overwrite headers
        metadata = dict(node.metadata)
        metadata["headers"] = norm_headers

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
    """Temporary harness to inspect CustomParser output."""

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

