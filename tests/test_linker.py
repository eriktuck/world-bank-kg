import pytest
import json
from unittest.mock import patch, MagicMock
from src.linker import Wikifier, num_tokens


@pytest.fixture
def wikifier():
    return Wikifier()


# ---------------------
# num_tokens
# ---------------------
@patch("src.linker.tiktoken.encoding_for_model")
def test_num_tokens_counts_correctly(mock_enc):
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = [1, 2, 3]
    mock_enc.return_value = mock_encoder

    result = num_tokens("hello", model="gpt-4")
    assert result == 3
    mock_enc.assert_called_once_with("gpt-4")


# ---------------------
# _sanitize_for_sparql
# ---------------------
def test_sanitize_for_sparql_removes_invalid_and_filters(wikifier):
    """Ensure the sanitizer removes invalid content and cleans up text."""
    # Valid case
    assert wikifier._sanitize_for_sparql("World Bank") == "World Bank"

    # Empty
    assert wikifier._sanitize_for_sparql("") is None

    # HTML fragments
    assert wikifier._sanitize_for_sparql("<div>hi</div>") is None

    # Numbers only
    assert wikifier._sanitize_for_sparql("12345") is None

    # Too short
    assert wikifier._sanitize_for_sparql("a") is None

    # Too long
    long_text = "A" * 300
    assert wikifier._sanitize_for_sparql(long_text) is None

    # Escaped quotes
    assert '\\"' in wikifier._sanitize_for_sparql('World "Bank"')


# ---------------------
# wikify
# ---------------------
@patch.object(Wikifier, "get_qid")
def test_wikify_includes_qid_and_safe_name(mock_get_qid, wikifier):
    """wikify should enrich entities with QIDs and sparql_safe strings."""
    mock_get_qid.side_effect = lambda x: "Q123" if "World" in x else None
    entities = [{"surface": "World Bank", "label": "ORG"}, {"surface": "!!!", "label": "MISC"}]

    result = wikifier.wikify(entities)

    assert isinstance(result, list)
    assert any(ent["qid"] == "Q123" for ent in result)
    assert all("sparql_safe" in ent for ent in result)
    mock_get_qid.assert_called()


@patch.object(Wikifier, "get_qid", return_value=None)
def test_wikify_skips_unsanitizable_entities(mock_get_qid, wikifier):
    """Entities that cannot be sanitized should be skipped."""
    entities = [{"surface": "<b>123</b>", "label": "MISC"}]
    result = wikifier.wikify(entities)
    assert result == []


# ---------------------
# wikify_from_llm
# ---------------------
@patch("src.linker.num_tokens", return_value=10)
def test_wikify_from_llm_batches_and_calls_wikify_batch(mock_num_tokens, wikifier):
    """Should batch entities properly and call _wikify_batch."""
    client = MagicMock()
    wikifier._wikify_batch = MagicMock(return_value=[("A", "B", "Q1", 0)])

    entities = [("World Bank", "ORG"), ("UNESCO", "ORG")]
    result = wikifier.wikify_from_llm(entities, client=client, model="mock-model")

    assert result == [("A", "B", "Q1", 0)]
    wikifier._wikify_batch.assert_called_once()
    mock_num_tokens.assert_called()


# ---------------------
# _wikify_batch
# ---------------------
def test__wikify_batch_parses_json_response(wikifier):
    """Ensure _wikify_batch constructs the right prompt and parses response JSON."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({"World Bank": "Q123"})))]
    mock_client.chat.completions.create.return_value = mock_response

    batch = [("World Bank", "ORG")]
    result = wikifier._wikify_batch(batch, mock_client, model="gpt-4")

    assert isinstance(result, dict)
    assert "World Bank" in result


# ---------------------
# get_qid
# ---------------------
@patch.object(Wikifier, "get_qid_via_exact_match", return_value="Q111")
def test_get_qid_returns_exact_match(mock_exact, wikifier):
    """If exact match exists, should return immediately."""
    result = wikifier.get_qid("World Bank")
    assert result == "Q111"
    mock_exact.assert_called_once()


@patch.object(Wikifier, "get_qid_via_exact_match", return_value=None)
@patch.object(Wikifier, "query_via_fuzzy_search", return_value="Q999")
def test_get_qid_falls_back_to_fuzzy(mock_fuzzy, mock_exact, wikifier):
    """Falls back to fuzzy search if exact fails."""
    result = wikifier.get_qid("UNESCO")
    assert result == "Q999"
    mock_fuzzy.assert_called_once()


@patch.object(Wikifier, "get_qid_via_exact_match", return_value=None)
@patch.object(Wikifier, "query_via_fuzzy_search", return_value=None)
def test_get_qid_returns_none_if_all_fail(mock_fuzzy, mock_exact, wikifier):
    """If both methods fail, return None."""
    result = wikifier.get_qid("Unknown Entity")
    assert result is None


# ---------------------
# get_qid_via_exact_match
# ---------------------
@patch("src.linker.SPARQLWrapper")
def test_get_qid_via_exact_match_returns_qid(mock_sparql_cls, wikifier):
    """Should parse SPARQL result and return QID."""
    mock_sparql = mock_sparql_cls.return_value
    mock_sparql.query.return_value.convert.return_value = {
        "results": {"bindings": [{"item": {"value": "http://www.wikidata.org/entity/Q789"}}]}
    }
    result = wikifier.get_qid_via_exact_match("World Bank")
    assert result == "Q789"


@patch("src.linker.SPARQLWrapper")
def test_get_qid_via_exact_match_handles_empty_results(mock_sparql_cls, wikifier):
    mock_sparql = mock_sparql_cls.return_value
    mock_sparql.query.return_value.convert.return_value = {"results": {"bindings": []}}
    result = wikifier.get_qid_via_exact_match("Unknown")
    assert result is None


# ---------------------
# query_via_fuzzy_search
# ---------------------
@patch("src.linker.requests.get")
def test_query_via_fuzzy_search_returns_qid(mock_get, wikifier):
    mock_response = MagicMock()
    mock_response.json.return_value = {"search": [{"id": "Q321"}]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    qid = wikifier.query_via_fuzzy_search("World Bank")
    assert qid == "Q321"


@patch("src.linker.requests.get")
def test_query_via_fuzzy_search_returns_none_if_empty(mock_get, wikifier):
    mock_response = MagicMock()
    mock_response.json.return_value = {"search": []}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    qid = wikifier.query_via_fuzzy_search("Random Entity")
    assert qid is None


# ---------------------
# query_via_sparql
# ---------------------
@patch("src.linker.SPARQLWrapper")
def test_query_via_sparql_returns_bindings(mock_sparql_cls, wikifier):
    mock_sparql = mock_sparql_cls.return_value
    mock_sparql.query.return_value.convert.return_value = {"results": {"bindings": [{"x": {"value": "Q1"}}]}}
    result = wikifier.query_via_sparql("SELECT ?x WHERE {}")
    assert isinstance(result, list)
    assert "x" in result[0]
