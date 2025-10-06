import pytest
import json
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from rdflib import URIRef, RDF, Literal

from src.graph import KnowledgeGraph, COUNTRY_PROPERTY_MAP


@pytest.fixture
def kg(tmp_path):
    """Initialize a KnowledgeGraph with temp TTL path."""
    kg = KnowledgeGraph(ttl_path=tmp_path / "test.ttl")
    kg.linker = MagicMock()  # prevent real SPARQL calls
    kg.cache = {"countries": {}, "entities": {}}
    return kg


def test_init_creates_empty_graph_and_namespaces(kg):
    """Graph initializes empty and namespaces bound."""
    assert not kg.loaded
    namespaces = dict(kg.g.namespaces())
    assert "schema" in namespaces

    ns_value = str(namespaces["schema"]).rstrip("/")
    assert ns_value in ("http://schema.org", "https://schema.org")



def test_get_cache_returns_existing(kg):
    result = kg._get_cache()
    assert isinstance(result, dict)
    assert "countries" in result


def test_write_cache_writes_json(tmp_path, kg):
    path = tmp_path / "wikidata_cache.json"
    with patch("builtins.open", mock_open()) as m:
        kg._write_cache({"a": 1})
        handle = m()
        handle.write.assert_called()


def test_sanitize_column_removes_accents_and_spaces(kg):
    s = pd.Series(["México DF", "South-Africa", None])
    result = kg._sanitize_column(s)
    assert "Mexico_DF" in list(result)
    assert "South_Africa" in list(result)


@patch("src.graph.requests.get")
def test_get_metadata_fetches_and_sanitizes(mock_get, tmp_path, kg):
    """Simulate metadata download and ensure DataFrame returned."""
    fake_json = {
        "documents": {
            "1": {
                "id": "1",
                "display_title": "Doc 1",
                "last_modified_date": "2025-01-01",
                "docty": "", "count": "", "trustfund": "",
                "trustfund_key": "", "projn": "", "projectid": "", "owner": ""
            },
            "2": {
                "id": "2",
                "display_title": "Doc 2",
                "last_modified_date": "2025-01-02",
                "docty": "", "count": "", "trustfund": "",
                "trustfund_key": "", "projn": "", "projectid": "", "owner": ""
            },
        }
    }
    mock_get.return_value.content = json.dumps(fake_json).encode()

    df = kg.get_metadata(max_pages=1)
    assert isinstance(df, pd.DataFrame)
    assert "display_title" in df.columns
    assert "docty" in df.columns


def test_create_new_class_adds_triples(kg):
    uri = kg._create_new_class("Document", "A document")
    triples = list(kg.g.triples((uri, None, None)))
    assert any(RDF.type in t for t in triples)
    assert any(isinstance(t[2], Literal) for t in triples)


def test_create_instance_adds_type_and_label(kg):
    class_uri = kg._create_new_class("Thing", "Thing")
    instance = kg._create_instance(class_uri, "inst1", "Instance 1")
    triples = list(kg.g.triples((instance, None, None)))
    assert (instance, RDF.type, class_uri) in kg.g
    assert any(isinstance(o, Literal) for (_, _, o) in triples)


def test_add_country_to_graph_adds_country_with_sameAs(kg):
    uri = kg.add_country_to_graph("http://wikidata.org/entity/Q30", "USA", "United States")
    triples = list(kg.g.triples((uri, None, None)))
    assert (uri, kg.schema.sameAs, URIRef("http://wikidata.org/entity/Q30")) in kg.g
    assert (uri, kg.schema.name, Literal("United States", lang="en")) in kg.g


def test_enrich_individual_country_uses_cache(kg):
    country_uri = URIRef("http://worldbank.example.org/country/USA")
    kg.cache["countries"]["Q30"] = [{"predicate": "p", "object": "o", "type": "literal"}]
    result = kg.enrich_individual_country("Q30", country_uri, COUNTRY_PROPERTY_MAP, kg.cache)
    assert "countries" in result
    assert (country_uri, URIRef("p"), Literal("o")) in kg.g


@patch("src.graph.Wikifier")
def test_populate_graph_with_countries_adds_each_country(MockWikifier, kg):
    kg.metadata = pd.DataFrame({"count": ["Mexico", "Canada"]})
    kg.linker.get_qid.side_effect = lambda x: "Q" + str(len(x))
    kg.populate_graph_with_countries()
    assert any("country" in str(s) for s in kg.g.subjects())
    assert kg.linker.get_qid.call_count == 2


def test_add_metadata_to_graph_creates_subclasses(kg):
    kg.metadata = pd.DataFrame({"owner": ["World Bank"], "id": ["123"], "display_title": ["Doc 1"]})
    kg.add_metadata_to_graph(
        uri_ref="document",
        literal="Document",
        values=["123"],
        ids={"123": "Doc 1"},
        primary_key=True,
        extra_columns=["owner"]
    )
    triples = list(kg.g.triples((None, kg.schema.identifier, None)))
    assert any("123" in str(o) for (_, _, o) in triples)


def test_add_entities_links_to_doc(kg):
    """Ensure entities are linked to the doc and properly typed."""
    entities = [
        {"surface": "World Bank", "label": "ORG", "qid": None, "rdf_safe": "World_Bank"}
    ]
    kg.add_entities("123", entities)
    subj = URIRef("http://worldbank.example.org/document/123")
    ent = URIRef("http://worldbank.example.org/entity/World_Bank")
    assert (subj, kg.schema.mentions, ent) in kg.g
    assert (ent, RDF.type, kg.schema.Thing) in kg.g


def test_get_document_ids_returns_identifiers(kg):
    doc_uri = URIRef("http://worldbank.example.org/document/123")
    kg.g.add((doc_uri, kg.schema.identifier, Literal("123")))
    result = kg.get_document_ids()
    assert result == ["123"]


def test_get_url_by_id_returns_url(kg):
    doc_uri = URIRef("http://worldbank.example.org/document/123")
    kg.g.add((doc_uri, kg.schema.identifier, Literal("123")))
    kg.g.add((doc_uri, kg.schema.url, Literal("http://example.com")))
    assert kg.get_url_by_id("123") == "http://example.com"
    assert kg.get_url_by_id("999") is None


@patch("src.graph.requests.get")
def test_load_or_build_rebuilds_when_requested(mock_get, tmp_path):
    """Verify load_or_build removes old TTL and calls build/save."""
    fake_ttl = tmp_path / "fake.ttl"
    fake_ttl.write_text("dummy")

    with patch.object(KnowledgeGraph, "build") as m_build, \
         patch.object(KnowledgeGraph, "save") as m_save:
        kg = KnowledgeGraph.load_or_build(ttl_path=fake_ttl, rebuild=True)
        m_build.assert_called_once()
        m_save.assert_called_once()
        assert isinstance(kg, KnowledgeGraph)
