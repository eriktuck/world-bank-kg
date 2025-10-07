from typing import List, Tuple, Optional, Any, Dict
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import requests
from openai import OpenAI
import json
import logging
import tiktoken
import re
from ratelimit import limits, sleep_and_retry

from src.prompts import entity_linker_prompt

logger = logging.getLogger(__name__)
ONE_SECOND = 1

def num_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

class Wikifier:
    def __init__(self):
        pass


    def _sanitize_for_sparql(self, entity: str) -> str | None:
        """Sanitize entity text for safe SPARQL queries and Turtle serialization."""
        if not entity:
            return None

        entity = entity.strip()

        # Remove braces, backslashes, and other LaTeX-like artifacts
        entity = re.sub(r"[{}\\]", "", entity)

        # Collapse multiple spaces
        entity = re.sub(r"\s+", " ", entity)

        # --- Filtering rules ---

        # Skip if looks like HTML/XML fragment
        if re.search(r"</?\w+>", entity):
            return None

        # Skip if all digits/punctuation
        if re.fullmatch(r"[\d\W]+", entity):
            return None

        # Must contain at least 2 alphanumeric chars
        if len(re.sub(r"[^A-Za-z0-9]", "", entity)) < 2:
            return None

        # Must contain at least one alphabetic character
        if not any(ch.isalpha() for ch in entity):
            return None

        # Skip absurdly short or long entities
        if len(entity) < 2 or len(entity) > 200:
            return None

        # Escape quotes for SPARQL safety
        entity = entity.replace('"', '\\"')

        return entity
        
    
    def wikify(self, entities: List[Dict]) -> List[Dict]:
        # TODO: Add cacheing for cross-document entity lookup
        entities_with_qid = []
        seen = set()

        for ent in entities:
            entity_name = ent.get("surface")
            entity_type = ent.get("label")

            # Skip if already seen
            key = (entity_name, entity_type)
            if key in seen:
                logger.debug(f"Skipping duplicate entity: {key}")
                continue
            seen.add(key)

            safe_name = self._sanitize_for_sparql(entity_name)
            if not safe_name:
                logger.debug(f"{entity_name} could not be sanitized")
                continue

            qid = self.get_qid(safe_name)
            if qid:
                logger.debug(f"{entity_name}: {qid}")
            else:
                logger.debug(f"{entity_name}: has no QID")

            entities_with_qid.append({
                "surface": entity_name,
                "label": entity_type,
                "sparql_safe": safe_name,
                "qid": qid,
            })

        return entities_with_qid
    
    
    def wikify_from_llm(
        self, 
        entities: List[Tuple[str, str]], 
        client: Optional[Any] = None,
        max_tokens: int = 8000,
        model: str = "gpt-5-mini-2025-08-07"
    ) -> List[Tuple[str, str, str, int]]:
        """THIS RETURNS NO RESULTS"""
        # TODO: parallelize
        if not client:
            client = OpenAI()

        all_results = []
        current_batch = []
        batch_tokens = num_tokens(entity_linker_prompt, model=model) + 500  # buffer for system msgs + response

        for ent in entities:
            candidate = f"{ent[0]} ({ent[1]})"
            ent_tokens = num_tokens(candidate, model=model)

            # if adding this entity would exceed the limit, send the current batch
            if batch_tokens + ent_tokens > max_tokens:
                all_results.extend(self._wikify_batch(current_batch, client, model))
                current_batch = []
                batch_tokens = num_tokens(entity_linker_prompt, model=model) + 500

            current_batch.append(ent)
            batch_tokens += ent_tokens

        # send last batch
        if current_batch:
            all_results.extend(self._wikify_batch(current_batch, client, model))

        return all_results


    def _wikify_batch(self, batch: List[Tuple[str, str]], client, model: str):
        entities_str = ", ".join(f"{ent[0]} ({ent[1]})" for ent in batch)
        prompt = entity_linker_prompt + "---\n\n" + entities_str
        
        logger.info(f"Prompt sent to LLM for {len(batch)} entities: {prompt[:250]} ... {prompt[250:]}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a wikifier that matches entities to WikiData entries by retrieving the entities QID."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0,
            response_format={"type": "json_object"}
        )

        logger.info("Raw response object: %s", response)
        logger.info("Raw message content: %s", response.choices[0].message.content)

        return json.loads(response.choices[0].message.content)


    @sleep_and_retry
    @limits(calls=1, period=ONE_SECOND)
    def get_qid(self, entity_name: str) -> str:
        exact_match_qid = self.get_qid_via_exact_match(entity_name)
        if exact_match_qid:
            return exact_match_qid

        fuzzy_match_qid = self.query_via_fuzzy_search(entity_name)
        if fuzzy_match_qid:
            return fuzzy_match_qid

        return None


    def get_qid_via_exact_match(self, entity_name: str) -> str:
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        query = f"""
            SELECT ?item WHERE {{
            ?item rdfs:label "{entity_name}"@en .
            }}
            LIMIT 1
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.addCustomHttpHeader("User-Agent", "WorldBankKGBot/1.0 (eriktuck@gmail.com)")
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            return results["results"]["bindings"][0]["item"]["value"].split("/")[-1]
        return None


    def query_via_fuzzy_search(self, entity_name: str) -> str:
        """Tries to find the QID via the Wikidata search API (fuzzy match). http://www.wikidata.org/entity/"""
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": entity_name,
            "language": "en",
            "format": "json",
            "type": "item"
        }
        headers = {
            "User-Agent": "WorldBankKGBot/1.0 (eriktuck@gmail.com)"
        }
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        search_results = response.json()

        if search_results['search']:
            qid = search_results['search'][0]['id']
            return qid
        return None
    

    def query_via_sparql(self, query: str) -> str:
        """Runs any SPARQL query."""
        # Set up SPARQL connection
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)
        sparql.addCustomHttpHeader("User-Agent", "WorldBankKGBot/1.0 (eriktuck@gmail.com)")
        
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results['results']['bindings']

        return bindings
    
        if bindings:
            first_binding = bindings[0]
            # Get the first variable name from the binding
            var_name = next(iter(first_binding))
            return first_binding[var_name]['value']
        return None


# def main_loop_from_notebook(countries):
#     # Main loop
#     for country in tqdm(countries):
#         if not isinstance(country, str):
#             continue

#         label = country.replace('_', ' ')
#         query = f"""
#             SELECT ?country WHERE {{
#                 ?country wdt:P31 wd:Q6256 .
#                 ?country (rdfs:label|skos:altLabel) "{label}"@en .
#             }}
#             LIMIT 1
#         """

#         try:
#             qid_uri = query_via_sparql(query)
#             if qid_uri:
#                 add_country_to_graph(qid_uri, country, label)
#             else:
#                 print(f"No exact match for: {label}. Trying fuzzy search...")
#                 try:
#                     qid_uri = query_via_fuzzy_search(label)
#                     if qid_uri:
#                         add_country_to_graph(qid_uri, country, label)
#                         print(f"Fuzzy match found: {label} → {qid_uri.split('/')[-1]}")
#                     else:
#                         print(f"No fuzzy match found for: {label}")
#                 except Exception as fuzzy_error:
#                     print(f"Fuzzy search failed for {label}: {fuzzy_error}")
#         except Exception as e:
#             print(f"SPARQL query failed for {label}: {e}")

def main():
    entity_linker = Wikifier()
    
    ent = "the World Bank"
    QID = entity_linker.get_qid(ent)

    print(QID)

if __name__ == "__main__":
    main()