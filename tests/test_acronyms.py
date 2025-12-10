import pytest
from unittest.mock import MagicMock, patch
from src.acronyms import AcronymExtractor
import types

@pytest.fixture
def extractor():
    return AcronymExtractor(doc_id="doc123")


@patch("src.acronyms.load_existing_index")
def test_get_acronym_section_returns_combined_text(mock_load_index, extractor):
    """Ensure acronym sections are concatenated correctly from retrieved nodes."""
    mock_index = MagicMock()
    mock_retriever = MagicMock()
    mock_node1 = MagicMock()
    mock_node2 = MagicMock()

    # Set values on the mock nodes
    mock_node1.score = 0.9
    mock_node2.score = 0.8
    mock_node1.node.text = "Abbreviations section text 1"
    mock_node2.node.text = "Abbreviations section text 2"

    mock_retriever.retrieve.return_value = [mock_node1, mock_node2]
    mock_index.as_retriever.return_value = mock_retriever
    mock_load_index.return_value = mock_index

    result = extractor._get_acronym_section()

    assert "Abbreviations section text 1" in result
    assert "Abbreviations section text 2" in result
    assert "---" in result  # combined with separator


@patch("src.acronyms.load_existing_index")
def test_get_acronym_section_warns_when_no_nodes(mock_load_index, extractor, caplog):
    """Log a warning when no acronym sections are retrieved."""
    mock_index = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index.as_retriever.return_value = mock_retriever
    mock_load_index.return_value = mock_index

    result = extractor._get_acronym_section()

    assert result == ""
    assert "No nodes retrieved" in caplog.text


def test_extract_inline_acronyms_collects_abbreviations_and_cleans(extractor):
    """Ensure inline abbreviations are captured and cleaned of HTML entities."""
    doc = MagicMock()
    abrv1 = MagicMock(text="AI")
    abrv1._.long_form.text = "Artificial&nbsp;Intelligence"
    doc._.abbreviations = [abrv1]

    cleaned = extractor._extract_inline_acronyms(doc)
    assert "AI" in cleaned

    # normalize non-breaking spaces before comparing
    cleaned_text = next(iter(cleaned.values()))
    assert cleaned_text.replace("\xa0", " ") == "Artificial Intelligence"


def test_merge_acronym_dicts_merges_without_conflicts(extractor):
    primary = {"AI": "Artificial Intelligence"}
    detected = {"ML": "Machine Learning"}
    merged = extractor.merge_acronym_dicts(primary, detected)
    assert merged == {"AI": "Artificial Intelligence", "ML": "Machine Learning"}


def test_get_all_entities_from_acronyms_combines_flipped_dicts(extractor):
    primary = {"AI": "Artificial Intelligence"}
    detected = {"ML": "Machine Learning"}
    entities = extractor.get_all_entities_from_acronyms(primary, detected)
    assert entities == {"Artificial Intelligence": "AI", "Machine Learning": "ML"}


def test_clean_acronyms_filters_and_decodes(extractor):
    dirty = {
        "AI": "Artificial&nbsp;Intelligence",
        "ml": "machine learning",  # lowercase should be filtered
    }
    cleaned = extractor.clean_acronyms(dirty)
    assert "AI" in cleaned
    assert "ml" not in cleaned
    assert cleaned["AI"].replace("\xa0", " ") == "Artificial Intelligence"


@patch.object(AcronymExtractor, "_get_acronym_section", return_value="Abbreviations section text")
@patch.object(AcronymExtractor, "_extract_acronyms_with_llm", return_value={"AI": "Artificial Intelligence"})
@patch.object(AcronymExtractor, "_extract_inline_acronyms", return_value={"ML": "Machine Learning"})
def test_extract_combines_and_returns_acronyms(mock_inline, mock_llm, mock_section, extractor):
    """End-to-end extraction should merge acronym sets and populate attributes."""
    acronyms = extractor.extract("Sample text")

    assert acronyms == {"AI": "Artificial Intelligence", "ML": "Machine Learning"}
    assert extractor.acronyms == acronyms
    assert extractor.entities == {"Artificial Intelligence": "AI", "Machine Learning": "ML"}
