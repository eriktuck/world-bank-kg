import requests
import json
import pandas as pd
import numpy as np
import unidecode
import os
import logging
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal
from rdflib.namespace import SKOS, XSD
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

from src.linker import Wikifier
from src.storage import load_existing_index

logger = logging.getLogger(__name__)

CACHE_FILE = "wikidata_cache.json"

COLUMN_TO_SCHEMA = {
    "id": "identifier",
    "display_title": "name",
    "last_modified_date": "dateModified",
    "pdfurl": "url",
    "author": "author",
    "year": "datePublished",
    "docty": "genre",  # TODO: add doc type as skos:Concept and link
    "owner": "creator"  # TODO: add owner as schema:Organization
}

COUNTRY_PROPERTY_MAP = {
    "continent": "P30",
    "form_of_government": "P122",
    "political_system": "P1387",
    "currency": "P38",
    "official_language": "P37",
    "population": "P1082",
    "gdp_per_capita": "P2299"
}

class KnowledgeGraph():
    def __init__(self, ttl_path='world-bank-kg.ttl'):
        self.ttl_path = Path(ttl_path)
        self.g = Graph()
        self.loaded = False

        if self.ttl_path.exists():
            logger.info(f"Loading KG from {self.ttl_path}")
            self.g.parse(self.ttl_path, format="turtle")
            self.loaded = True
        else:
            logger.info("Initializing KG.")

        self.schema = Namespace("http://schema.org/")
        self.wd = Namespace('http://www.wikidata.org/entity/')
        self.ex = Namespace("http://worldbank.example.org/")
        self.url = 'https://search.worldbank.org/api/v2/wds'  # TODO: update to v3
        self.params = {
            'format': 'json',
            'docty': 'Project Appraisal Document',
            'qterm': 'wind turbine',
            'fl': ','.join(['id', 'display_title', 'count', 'trustfund', 'trustfund_key', 'projn', 'projectid', 'display_title', 'owner', 'pdfurl', 'year', 'last_modified_date', 'docty']),
            'rows': 20,
            'page':1
        }

        self.prefixes = {
            'schema': self.schema,
            'wd': self.wd,
            'ex': self.ex,
            'skos': SKOS,
            'xsd': XSD
        }
        for p, ns in self.prefixes.items():
            self.g.bind(p, ns)

        # Add identifier properties
        self.g.add((self.schema.identifier, RDF.type, RDF.Property))
        self.g.add((self.schema.identifier, RDFS.label, Literal("Identifier")))

        self.linker = Wikifier()
        self.metadata = None
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {
                "countries": {},
                "entities": {}
            }


    def __repr__(self):
        triples_by_subject = defaultdict(list)

        for s, p, o in self.g:
            triples_by_subject[s].append((p, o))

        lines = []
        for subject, po_list in triples_by_subject.items():
            lines.append(f"\n🔹 {subject}")
            for predicate, obj in po_list:
                lines.append(f"  ↳ {predicate} → {obj}")

        repr_str = "\n".join(lines)

        logger.debug("Graph contents:\n%s", repr_str)

        return repr_str


    def _get_cache(self):
        if self.cache:
            return self.cache
    

    def _write_cache(self, cache: Dict, write_type: str = 'w'):
        with open(CACHE_FILE, write_type) as f:
            json.dump(cache, f, indent=2)


    def _sanitize_column(self, series: pd.Series) -> pd.Series:
        """Clean strings for use in RDF identifiers (URIs or keys)."""
        return (series.astype(str)
                    .map(unidecode.unidecode) # remove accents
                    .str.replace(r'\s+', '_', regex=True) # replace spaces
                    .str.replace("-", "_") # replace hyphens
                    .str.strip("_")  # remove trailing underscore
                    .replace('nan', np.nan)  # return nan
        )
    

    def get_metadata(self, max_pages: int = 1):
        metadata_list = []

        # Reset page index
        self.params['page'] = 1

        for _ in range(max_pages):
            response = requests.get(self.url, params=self.params)
            response.raise_for_status()
            data = json.loads(response.content)

            documents = data.get("documents", {})
            if not documents:
                logger.warning("No documents found in API response.")
                break

            for key, metadata in documents.items():
                # Skip 'facets' and non-dict entries
                if not isinstance(metadata, dict):
                    logger.debug(f"Skipping non-dict entry at key {key}")
                    continue

                # Skip malformed docs with no 'id'
                if "id" not in metadata:
                    logger.debug(f"Skipping non-document entry: {key}")
                    continue

                metadata_list.append(metadata)

            self.params["page"] += 1

        df = pd.DataFrame(metadata_list)

        # Drop older versions of same document
        df = (
            df.assign(last_modified_date=pd.to_datetime(df["last_modified_date"]))
            .sort_values("last_modified_date", ascending=False)
            .drop_duplicates(subset="id", keep="first")
        )

        # TODO: new classes: theme, docty, owner
        columns_to_sanitize = [
            'docty', 'count', 'trustfund', 'trustfund_key',
            'projn', 'projectid', 'display_title', 'owner']
        
        df[columns_to_sanitize] = df[columns_to_sanitize].apply(
            self._sanitize_column
        )

        df.to_csv('output/raw.csv', index=False)

        self.metadata = df

        return self.metadata
    

    def _create_new_class(
            self, 
            uri_ref: str, 
            literal: str, 
            parent: Optional[URIRef] = None
        ) -> URIRef:
        """
        Create a new class in the custom namespace and link it to schema:CreativeWork by default.

        Parameters
        ----------
        uri_ref : str
            The local name for the new class (e.g., "Document").
        literal : str
            Human-readable label for the class.
        parent : URIRef, optional
            A superclass to link against. Defaults to schema:CreativeWork.
        """
        newClass = URIRef(self.ex + uri_ref)
        parentClass = parent if parent is not None else self.schema.Thing

        self.g.add((newClass, RDF.type, RDFS.Class))
        self.g.add((newClass, RDFS.subClassOf, parentClass))
        self.g.add((newClass, RDFS.label, Literal(literal, lang='en')))

        return newClass


    def _create_instance(self, class_uri: URIRef, uri_ref: str, label: str) -> URIRef:
        """Create a new instance (individual) of a given class."""
        instance_uri = URIRef(self.ex + uri_ref)
        self.g.add((instance_uri, RDF.type, class_uri))
        
        if label:
            self.g.add((instance_uri, self.schema.name, Literal(label, lang='en')))

        return instance_uri
    

    def _create_new_subclass(
            self, superClass: URIRef, uri_ref: str, value: str
        ) -> URIRef:
        """Create a subclass under a given superclass."""
        subclass = URIRef(uri_ref)
        self.g.add((subclass, RDF.type, RDFS.Class))
        self.g.add((subclass, RDFS.subClassOf, superClass))
        self.g.add((subclass, RDFS.label, Literal(value, lang='en')))

        return subclass
    

    def add_country_to_graph(self, qid_uri: str, country: str, label: str):
        """Adds a country instance to the RDF graph."""
        country_key = str(country).strip()
        local_uri = self.ex[f"country/{country_key}"]

        wikidata_uri = URIRef(qid_uri) if qid_uri else None

        self.g.add((local_uri, RDF.type, self.schema.Country))
        self.g.add((local_uri, self.schema.name, Literal(label.strip(), lang='en')))
        
        if wikidata_uri:
            self.g.add((local_uri, self.schema.sameAs, wikidata_uri))

        return local_uri

    
    def populate_graph_with_countries(self):
        logger.debug(self.metadata['count'].dropna().unique())
        countries = self.metadata['count'].dropna().unique()
        for country in tqdm(countries):
            country_label = country.replace('_', ' ')
            qid = self.linker.get_qid(country_label)
            if qid:
                qid_uri = f"http://www.wikidata.org/entity/{qid}"
            else:
                qid_uri = None    
            logger.debug(f"Added {country_label} with QID {qid}")
            
            self.add_country_to_graph(qid_uri, country, country_label)
            

    def enrich_individual_country(
            self, qid: str, country_uri: URIRef, properties: dict,
            cache: Dict) -> Dict:
        """
        Enriches a country node in the RDF graph with contextual properties from Wikidata.

        Caches enrichment data and reuses the cached triples to avoid redundant queries.

        Parameters:
            qid (str): The Wikidata QID of the country (e.g., "Q43" for Turkey).
            country_uri (rdflib.term.URIRef): The URI of the country node in the local knowledge graph.
            properties (dict): A dictionary like {'property': 'pid'}

        Side Effects:
            - Adds new triples to the global RDF graph `g`.
            - Updates the global cache `cache` with newly retrieved triples.

        Raises:
            Logs warning if the SPARQL query fails or returns unexpected results.
        """

        if qid in cache["countries"]:
            for triple in cache['countries'][qid]:
                subj = country_uri
                pred = URIRef(triple["predicate"])
                obj = URIRef(triple["object"]) if triple["type"] == "uri" else Literal(triple["object"])
                self.g.add((subj, pred, obj))
            return cache

        query = f"""
            SELECT ?property ?value ?valueLabel WHERE {{
                VALUES ?property {{ { ' '.join(f'wdt:{pid}' for pid in properties.values()) } }}
                wd:{qid} ?property ?value .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
        """
        try:
            cache_triples = []
            bindings = self.linker.query_via_sparql(query)
            for result in bindings:
                prop_uri = result['property']['value']
                value = result['value']['value']
                label = result.get('valueLabel', {}).get('value', None)
                pred = URIRef(prop_uri)
            
                # Use wikidata URI if object is an entity
                if value.startswith("http://www.wikidata.org/entity"):
                    obj = URIRef(value)
                    obj_type = "uri"
                
                    # Check if we've already added a label for this URIRef
                    if label and label not in cache["entities"]:
                        self.g.add((obj, self.schema.name, Literal(label, lang="en")))
                        cache["entities"][label] = value

                else:
                    obj = Literal(value)
                    obj_type = "literal"
                
                self.g.add((country_uri, pred, obj))
                cache_triples.append({
                    "predicate": str(prop_uri),
                    "object": str(value),
                    "type": obj_type
                })
                
            # Save to cache
            cache["countries"][qid] = cache_triples

            return cache

        except Exception as e:
            logger.warning(f"Error enriching {qid}: {e}")

    
    def enrich_all_country_data(self):
        cache = self._get_cache()
        for country_uri in self.g.subjects(RDF.type, self.schema.Country):
        
            # Look for schema:sameAs link to Wikidata
            qid_uri = next(self.g.objects(country_uri, self.schema.sameAs), None)

            if qid_uri and "wikidata.org/entity/" in str(qid_uri):
                # Extract the QID from the URI
                qid = str(qid_uri).split("/")[-1]
                new_cache = self.enrich_individual_country(
                    qid, country_uri, COUNTRY_PROPERTY_MAP, cache)
                # TODO cache with flush atomically outside loop
                self._write_cache(new_cache)
            else:
                print(f"No Wikidata QID found for {country_uri}, skipping...")
        

    def add_countries(self):
        self.populate_graph_with_countries()
        self.enrich_all_country_data()

    
    def add_metadata_to_graph(
            self, 
            uri_ref: URIRef, 
            literal: str, 
            values = List,
            query_template: Optional[str] = None,
            ids: Optional[Dict[str, str]] = None,
            primary_key: bool = False,
            extra_columns: Optional[List[str]] = None,
            parent_class: Optional[URIRef] = None
            ):
        """
        Generic method to add metadata from a column into the graph.

        Parameters
        ----------
        uri_ref : str
            Base URIRef suffix for the new parent class.
        literal : str
            Human-readable label for the parent class.
        values: list
            List of values to add as subclasses of the parent class.
        query_template : str, optional
            A SPARQL query template with `{value}` placeholder for Wikidata lookups.
        ids : dict, optional
            Mapping from value -> identifier (used for trustfunds or similar).
        primary_key : bool
            If True, use identifier (id) as the URI for the resource.
        extra_columns : list, optional
            List of metadata DataFrame columns to attach as schema:* properties.
        """
        newClass = self._create_new_class(uri_ref, literal, parent_class)

        # Add ID property to graph if provided
        id_property = self.schema.identifier
        self.g.add((id_property, RDFS.domain, newClass))

        # Add domains for any extra properties used
        if extra_columns:
            for col in extra_columns:
                schema_prop_name = COLUMN_TO_SCHEMA.get(col, col)
                if hasattr(self.schema, schema_prop_name):
                    prop = getattr(self.schema, schema_prop_name)
                    self.g.add((prop, RDFS.domain, newClass))

        # Populate graph with subclasses
        for value in values:
            if not isinstance(value, str):
                continue
            
            # Create instance of newClass and use ID in URI
            if primary_key:
                identifier = value.strip()
                label = ids.get(identifier, identifier) if ids else identifier
                instance_uri = f"{uri_ref}/{identifier}"
                resource = self._create_instance(newClass, instance_uri, label)
                self.g.add((resource, id_property, Literal(identifier)))

            # Create subclass of newClass and use value in URI
            else: 
                uri_fragment = value.replace(" ", "_").replace("-", "_")
                subclass_uri = f"{uri_ref}/{uri_fragment}"
                resource = self._create_new_subclass(newClass, self.ex + subclass_uri, value)
                if ids and value in ids:
                    identifier = ids[value]
                    self.g.add((resource, id_property, Literal(identifier)))

            # Attach extra fields for both instances and classes
            if extra_columns and self.metadata is not None:
                row = self.metadata[self.metadata.isin([value]).any(axis=1)]
                if not row.empty:
                    row = row.iloc[0]
                    for col in extra_columns:
                        if col in row and pd.notna(row[col]):
                            schema_prop_name = COLUMN_TO_SCHEMA.get(col, col)
                            if hasattr(self.schema, schema_prop_name):
                                prop = getattr(self.schema, schema_prop_name)
                                self.g.add((resource, prop, Literal(str(row[col]))))

            # Query Wikidata and link Wikidata URI
            if query_template:
                safe_value = value.replace('"', '\\"')
                query = query_template.format(value=safe_value)
                bindings = self.linker.query_via_sparql(query)
                
                if bindings:
                    wd_class = bindings[0]['class']['value']
                    wd_uri = URIRef(wd_class)
                    self.g.add((resource, self.schema.wd_URI, wd_uri))


    def add_world_bank_documents(
            self, 
            column: str = 'display_title', 
            id_column: str = 'id',
            extra_columns: Optional[List[str]] = None
        ):
        """ Adds all world bank documents as ID-driven instances with extra metadata."""
        uri_ref = "document"
        literal = "A document produced and written for the World Bank."
        parent_class = self.schema.CreativeWork

        doc_dict = {}
        
        for _, row in self.metadata[[column, id_column]].dropna().iterrows():
            title = str(row[column]).strip()
            doc_id = str(row[id_column]).strip()
            doc_dict[doc_id] = title
        
        values = list(doc_dict.keys())

        self.add_metadata_to_graph(
            uri_ref=uri_ref,
            literal=literal,
            values=values,
            ids=doc_dict,
            primary_key=True,
            extra_columns=extra_columns,
            parent_class=parent_class,
        )


    def _add_id_name_entities(self, name_col: str, id_col: str, uri_ref: str, literal: str):
        """Generic helper for entities stored as ID + Name pairs (possibly comma-separated)."""

        entity_dict = {}
        for _, row in self.metadata[[name_col, id_col]].dropna().iterrows():
            ids = [i.strip() for i in str(row[id_col]).split(',')  if i.strip()]
            names = [n.strip() for n in str(row[name_col]).split(',') if n.strip()]
            
            # Zip protects against mismatch length — ignores extras
            for ent_id, ent_name in zip(ids, names):
                entity_dict[ent_id] = ent_name

        values = list(entity_dict.keys())

        self.add_metadata_to_graph(
            uri_ref=uri_ref,
            literal=literal,
            values=values,
            ids=entity_dict,
            primary_key=True
        )


    def add_trustfunds(self, name_col: str = 'trustfund', id_col: str = 'trustfund_key'):
        """Adds all trustfunds from metadata. Trustfunds can be comma-separated"""
        self._add_id_name_entities(name_col, id_col, "trustfund", "World Bank Trustfund")


    def add_projects(self, name_col:str = 'projn', id_col: str = 'projectid'):
        """Adds all projects from metadata. Projects can be comma-separated"""
        self._add_id_name_entities(name_col, id_col, "project", "World Bank Project")

    
    def _link_documents_to_entities(
        self,
        doc_col: str,
        entity_col: str,
        uri_ref: str,
        predicate: URIRef,
        is_multi_entity: bool = True
    ):
        """
        Generic helper to link documents to entities via a given predicate.

        Parameters
        ----------
        doc_col : str
            Column containing document IDs (usually 'id').
        entity_col : str
            Column containing entity IDs (e.g. 'count', 'projectid', 'trustfund_key').
        uri_ref : str
            URI namespace suffix for the entity (e.g. 'country', 'project', 'trustfund').
        predicate : rdflib.term.URIRef
            The property to use for linking (e.g. schema:countryOfOrigin, schema:isPartOf, schema:funder).
        is_multi_entity: bool
            True if the column can contain comma-separated entities (e.g., 'Project A, Project B')
        """
        if self.metadata is None:
            logger.warning("Metadata is not loaded; run get_metadata() first.")
            return

        for _, row in self.metadata.iterrows():
            doc_id = str(row.get(doc_col))
            entities = row.get(entity_col)

            if pd.isna(doc_id) or pd.isna(entities):
                continue

            doc_uri = self.ex[f"document/{doc_id}"]

            # Entities can be comma-separated
            if is_multi_entity:
                ent_list = [e.strip() for e in str(entities).split(",") if e.strip()]
            else:
                ent_list = [str(entities).strip()]
            
            for ent_id in ent_list:
                if not ent_id:
                    continue

                ent_uri = self.ex[f"{uri_ref}/{ent_id}"]

                # Ensure entity exists (should have been created by add_* already)
                if not list(self.g.triples((ent_uri, None, None))):
                    logger.debug(f"Adding fallback {uri_ref} {ent_id} for doc {doc_id}")
                    self.g.add((ent_uri, RDF.type, self.ex[uri_ref]))
                    self.g.add((ent_uri, self.schema.identifier, Literal(ent_id)))

                # Link doc → entity
                self.g.add((doc_uri, predicate, ent_uri))
                
    def link_documents_to_countries(self):
        """
        Link each document in metadata to its associated country
        using schema:countryOfOrigin.
        """
        self._link_documents_to_entities(
            doc_col="id",
            entity_col="count",
            uri_ref="country",
            predicate=self.schema.countryOfOrigin,
            is_multi_entity=False
        )

    
    def link_documents_to_projects(self):
        """
        Link each document in metadata to its associated project(s)
        using schema:isPartOf.
        """
        self._link_documents_to_entities(
            doc_col="id",
            entity_col="projectid",
            uri_ref="project",
            predicate=self.schema.isPartOf
        )

    
    def link_documents_to_trustfunds(self):
        """
        Link each document in metadata to its associated project(s)
        using schema:funder.
        """
        self._link_documents_to_entities(
            doc_col="id",
            entity_col="trustfund_key",
            uri_ref="trustfund",
            predicate=self.schema.funder
        )

    
    def build(self):
        """Builds knowledge graph from metadata."""
        self.get_metadata()
        self.add_countries()
        self.add_world_bank_documents(
            extra_columns=['pdfurl', 'last_modified_date', 'docty', 'owner'])
        self.add_trustfunds()
        self.add_projects()
        self.link_documents_to_countries()
        self.link_documents_to_projects()
        self.link_documents_to_trustfunds()
    

    def save(self):
        """Save graph to file. Ensure format and file extension are compatible."""
        self.g.serialize(
            destination=self.ttl_path, 
            format='turtle', 
            prefixes=self.prefixes, 
            encoding='utf-8'
        )
        logger.info(f"Knowledge graph saved to {self.ttl_path}")


    def get_document_ids(self) -> List[str]:
        """Return a list of document nodes in the graph."""
        ids = []
        for s, p, o in self.g.triples((None, self.schema.identifier, None)):
            ids.append(str(o))
        return ids
    

    def get_url_by_id(self, doc_id: str) -> Optional[str]:
        """
        Given a document ID (as a string), return the associated URL if it exists.
        """
        for s, p, o in self.g.triples((None, self.schema.identifier, None)):
            if str(o) == doc_id:
                # Look for the URL of this subject
                for _, _, url in self.g.triples((s, self.schema.url, None)):
                    return str(url)
        return None
    

    def add_entities(self, doc_id: str, entities: List[Dict]):
        """
        Add extracted entities to the graph, linking them to the given document.
        entities: dict [surface_form, label, qid_or_none]
        """
        doc_uri = self.ex[f"document/{doc_id}"]

        for ent in entities:
            surface = ent["surface"]
            label = ent["label"]
            qid = ent["qid"]
            safe_id = ent["rdf_safe"]

            if not surface or not safe_id:
                continue    

            entity_uri = self.wd[qid] if qid else self.ex[f"entity/{safe_id}"]

            try:
                # Add entity node
                self.g.add((entity_uri, RDF.type, self.schema.Thing))
                self.g.add((entity_uri, self.schema.name, Literal(surface)))
            
                # Add label as type info
                if label:
                    self.g.add((entity_uri, self.schema.additionalType, Literal(label)))

                # Link doc with entity
                self.g.add((doc_uri, self.schema.mentions, entity_uri))
            
            except Exception as e:
                logger.warning(f'Skipping bad entity {surface}: {e}')

    
    
    def add_text_chunks(self, doc_id: str):
        """
        Add text chunks as nodes in the graph and link them to existing entities.

        Each chunk node is linked to:
        - Its parent document (:isPartOf)
        - Entities it mentions (:mentions), using existing entity URIs
        """
        storage_context = load_existing_index().storage_context
        docstore = storage_context.docstore

        info = docstore.get_ref_doc_info(doc_id)
        if not info:
            logger.warning(f"No nodes found for doc_id={doc_id}")
            return

        # enrich metadata in docstore
        nodes = docstore.get_nodes(info.node_ids)
        doc_uri = self.ex[f"document/{doc_id}"]

        for i, node in enumerate(nodes):
            node_id = getattr(node, "node_id", f"{doc_id}_chunk_{i}")
            node_uri = self.ex[f"chunk/{node_id}"]

            logger.debug(node.metadata)

            # Add chunk node
            self.g.add((node_uri, RDF.type, self.schema.TextObject))
            self.g.add((node_uri, self.schema.text, Literal(node.text)))
            self.g.add((node_uri, self.schema.isPartOf, doc_uri))

            logger.debug(f"Entities found: {node.metadata.get('entities', [])}")

            for ent in node.metadata.get('entities', []):
                
                qid = ent.get('qid')
                rdf_safe = ent.get("rdf_safe")
                
                if qid:
                    ent_uri = self.wd[qid]
                elif rdf_safe:
                    ent_uri = self.ex[f"entity/{rdf_safe}"]
                else:
                    continue

                # Only link if entity URI already exists in graph
                if (ent_uri, RDF.type, None) in self.g:
                    self.g.add((node_uri, self.schema.mentions, ent_uri))
                else:
                    logger.debug(f"Entity {ent_uri} not found in graph; skipping link.")

        logger.info(f"Added {len(nodes)} chunks for document {doc_id}.")


    @classmethod
    def load_or_build(
        cls, 
        ttl_path="world-bank-kg.ttl", 
        rebuild: bool = False
    ):
        """
        Load existing KG if available, otherwise build a new one.
        If rebuild=True, overwrite any existing file and force a rebuild.
        """
        ttl_file = Path(ttl_path)

        if rebuild: 
            if ttl_file.exists():
                logger.warning(f"Rebuilding KG, removing existing file at {ttl_file}")
                ttl_file.unlink()
        
            cache_path = Path(CACHE_FILE)
            if cache_path.exists():
                logger.warning(f"Removing cache file at {cache_path}")
                cache_path.unlink()

        kg = cls(ttl_path)
        logger.info(f"Loaded KG with {len(kg.g)} triples")
        
        if rebuild or not kg.loaded:
            kg.build()
            kg.save()

        return kg


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger.info("Building Knowledge Graph...")
    kg = KnowledgeGraph.load_or_build('world-bank-kg.ttl', rebuild=False)

    # kg.add_text_chunks(doc_id=str(10170637))
    

    # print(kg)

    kg.save()


if __name__ == '__main__':
    main()
    


