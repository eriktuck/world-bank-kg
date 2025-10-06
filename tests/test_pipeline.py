import pytest
from unittest.mock import patch, MagicMock
from src.pipeline import DocumentPipeline


@pytest.fixture
def mock_spacy_model():
    """Create a fake SpaCy NLP model with expected pipes."""
    mock_nlp = MagicMock(name="nlp")
    mock_nlp.pipe_names = []
    mock_entity_ruler = MagicMock(name="entity_ruler")
    mock_nlp.add_pipe.return_value = mock_entity_ruler
    return mock_nlp


@pytest.fixture
def mock_doc():
    """Fake SpaCy Doc with entity spans."""
    doc = MagicMock(name="doc")
    ent1 = MagicMock(text="World Bank", label_="ORG")
    ent2 = MagicMock(text="AI", label_="ACRONYM")
    doc.ents = [ent1, ent2]
    return doc


@patch("src.pipeline.spacy.load")
def test_init_adds_pipes(mock_spacy_load, mock_spacy_model):
    """Ensure abbreviation_detector and entity_ruler are added on init."""
    mock_spacy_load.return_value = mock_spacy_model

    pipeline = DocumentPipeline(file_id="doc123")

    # Abbreviation detector added first
    mock_spacy_model.add_pipe.assert_any_call("abbreviation_detector", first=True)
    # Entity ruler added before NER
    mock_spacy_model.add_pipe.assert_any_call("entity_ruler", before="ner")

    assert hasattr(pipeline, "entity_ruler")
    assert isinstance(pipeline.acronym_extractor, object)
    assert isinstance(pipeline.entity_extractor, object)


@patch("src.pipeline.Wikifier")
@patch("src.pipeline.EntityExtractor")
@patch("src.pipeline.AcronymExtractor")
@patch("src.pipeline.spacy.load")
def test_process_invokes_components(mock_spacy_load, MockAcronymExtractor, MockEntityExtractor, MockWikifier, mock_spacy_model, mock_doc):
    """Verify the process method orchestrates the full pipeline correctly."""
    mock_spacy_load.return_value = mock_spacy_model
    mock_spacy_model.return_value = mock_doc

    # Mock AcronymExtractor
    mock_acronym_extractor = MockAcronymExtractor.return_value
    mock_acronym_extractor.extract.return_value = {"AI": "Artificial Intelligence"}

    # Mock EntityExtractor
    mock_entity_extractor = MockEntityExtractor.return_value
    mock_entity_extractor.apply_entity_ruler.return_value = MagicMock(name="ruler_doc")
    mock_entity_extractor.collect_entities.return_value = [
        {"surface": "World Bank", "label": "ORG"},
        {"surface": "AI", "label": "ACRONYM"},
    ]
    mock_entity_extractor._normalize_entities.return_value = [
        {"surface": "World Bank", "label": "ORG", "rdf_safe": "World_Bank"},
        {"surface": "AI", "label": "ACRONYM", "rdf_safe": "AI"},
    ]

    # Mock Wikifier
    mock_wikifier = MockWikifier.return_value
    mock_wikifier.wikify.return_value = [
        {"surface": "World Bank", "label": "ORG", "qid": "Q784"},
        {"surface": "AI", "label": "ACRONYM", "qid": "Q11660"},
    ]

    pipeline = DocumentPipeline(file_id="doc999")
    acronyms, entities = pipeline.process("AI stands for Artificial Intelligence")

    # Check that apply_entity_ruler and collect_entities were called properly
    mock_entity_extractor.apply_entity_ruler.assert_called_once_with(
        pipeline.entity_ruler, mock_doc
    )
    mock_entity_extractor.collect_entities.assert_called_once_with(
        mock_entity_extractor.apply_entity_ruler.return_value
    )

    # Check data flow integrity
    mock_wikifier.wikify.assert_called_once()
    mock_entity_extractor._normalize_entities.assert_called_once()
    assert acronyms == {"AI": "Artificial Intelligence"}
    assert {"surface": "World Bank", "label": "ORG", "rdf_safe": "World_Bank"} in entities



@patch("src.pipeline.DocumentPipeline.process")
@patch("src.pipeline.Reader")
def test_run_reads_and_processes(mock_reader, mock_process):
    """Ensure run() reads markdown and calls process()."""
    mock_reader.return_value = mock_reader
    mock_reader.return_value.get_markdown.return_value = "Sample markdown"
    mock_process.return_value = (
        {"AI": "Artificial Intelligence"},
        [{"surface": "AI", "label": "ACRONYM"}],
    )

    pipeline = DocumentPipeline(file_id="doc111")
    result = pipeline.run()

    # Ensure Reader and process used correctly
    mock_reader.return_value.get_markdown.assert_called_once_with("doc111")
    mock_process.assert_called_once_with("Sample markdown")

    # Check output structure
    assert result["doc_id"] == "doc111"
    assert "acronyms" in result
    assert "entities" in result
    assert result["acronyms"]["AI"] == "Artificial Intelligence"
