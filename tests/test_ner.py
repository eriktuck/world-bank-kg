import pytest
import logging
from unittest.mock import MagicMock
from spacy.tokens import Doc, Span

from src.ner import EntityExtractor, EXCLUDED_ENTS


@pytest.fixture
def extractor():
    return EntityExtractor()


def test_sanitize_for_rdflib_returns_safe_value(extractor):
    entity = "World Bank"
    safe = extractor._sanitize_for_rdflib(entity)
    assert "World_Bank" in safe
    assert len(safe) <= 100


def test_sanitize_for_rdflib_returns_none_on_invalid_uri(extractor):
    result = extractor._sanitize_for_rdflib("")
    assert result is None


def test_normalize_entities_adds_rdf_safe(extractor):
    entities = [
        {"surface": "World Bank", "label": "ORG", "qid": None},
        {"surface": "UNESCO", "label": "ORG", "qid": "Q123"}
    ]
    normalized = extractor._normalize_entities(entities)

    assert all("rdf_safe" in e for e in normalized)
    assert normalized[0]["rdf_safe"].startswith("World_Bank")
    assert normalized[1]["rdf_safe"] == "Q123"


def test_add_acronym_patterns_adds_both_acronym_and_expanded(extractor, caplog):
    mock_entity_ruler = MagicMock()
    acronyms = {"STEP": "Systematic Tracking and Exchanges in Procurement"}
    caplog.set_level(logging.DEBUG)

    extractor.add_acronym_patterns(mock_entity_ruler, acronyms)

    # Expected pattern structure
    expected_patterns = [
        {
            "label": "ACRONYM",
            "pattern": [{"LOWER": "step"}],
            "id": "STEP"
        },
        {
            "label": "ACRONYM_EXPANDED",
            "pattern": [
                {"LOWER": "systematic"},
                {"LOWER": "tracking"},
                {"LOWER": "and"},
                {"LOWER": "exchanges"},
                {"LOWER": "in"},
                {"LOWER": "procurement"},
            ],
            "id": "Systematic Tracking and Exchanges in Procurement"
        }
    ]

    mock_entity_ruler.add_patterns.assert_called_once_with(expected_patterns)
    assert "Added 2 acronym + expanded patterns" in caplog.text


def test_add_acronym_patterns_handles_empty_dict(extractor, caplog):
    mock_entity_ruler = MagicMock()
    caplog.set_level(logging.DEBUG)

    extractor.add_acronym_patterns(mock_entity_ruler, {})
    mock_entity_ruler.add_patterns.assert_not_called()
    assert "Added" not in caplog.text


def test_apply_entity_ruler_returns_modified_doc(extractor):
    mock_entity_ruler = MagicMock()
    mock_doc = MagicMock()
    mock_entity_ruler.return_value = mock_doc

    result = extractor.apply_entity_ruler(mock_entity_ruler, mock_doc)
    assert result == mock_doc
    mock_entity_ruler.assert_called_once_with(mock_doc)


def test_collect_entities_filters_excluded_labels(extractor):
    mock_doc = MagicMock()
    ent1 = MagicMock(text="World Bank", label_="ORG")
    ent2 = MagicMock(text="2020", label_="DATE")
    ent3 = MagicMock(text="STEP", label_="ACRONYM")
    ent4 = MagicMock(text="Systematic Tracking and Exchanges in Procurement", label_="ACRONYM_EXPANDED")

    mock_doc.ents = [ent1, ent2, ent3, ent4]

    result = extractor.collect_entities(mock_doc)

    # Verify excluded entity filtered
    assert all(e["label"] not in EXCLUDED_ENTS for e in result)

    # Verify both acronym types captured
    labels = [e["label"] for e in result]
    assert "ACRONYM" in labels
    assert "ACRONYM_EXPANDED" in labels
    assert {"surface": "Systematic Tracking and Exchanges in Procurement", "label": "ACRONYM_EXPANDED"} in result


def test_collect_entities_empty_doc_returns_empty_list(extractor):
    mock_doc = MagicMock()
    mock_doc.ents = []
    result = extractor.collect_entities(mock_doc)
    assert result == []
