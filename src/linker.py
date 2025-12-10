from typing import List, Tuple, Optional, Any, Dict
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import requests
from openai import OpenAI
import json
import logging
from ratelimit import limits, sleep_and_retry
import os

from src.prompts import entity_linker_prompt
from src.utils import sanitize_for_sparql, num_tokens

logger = logging.getLogger(__name__)

ONE_SECOND = 1

EMAIL_ADDRESS = "eriktuck@gmail.com"

CACHE_FILE = 'cache/wikidata_cache.json'  # TODO: move centrally

TYPE_QID_MAP = {
    "CARDINAL": ["wd:Q11229"],                # number
    "DATE": ["wd:Q205892"],                   # point in time
    "EVENT": ["wd:Q1656682"],                 # event
    "FAC": ["wd:Q13226383", "wd:Q17334923"],  # facility / infrastructure
    "GPE": ["wd:Q6256", "wd:Q3624078"],       # country / sovereign state
    "LANGUAGE": ["wd:Q34770"],                # human language
    "LAW": ["wd:Q7748"],                      # law / legal act
    "LOC": ["wd:Q2221906"],                   # geographic location
    "MONEY": ["wd:Q1368"],                    # money
    "NORP": ["wd:Q16334295", "wd:Q9174"],     # demographic group, ethnic group
    "ORDINAL": ["wd:Q628523"],                # ordinal number
    "ORG": ["wd:Q43229"],                     # organization
    "PERCENT": ["wd:Q11229"],                 # number (percentage)
    "PERSON": ["wd:Q5"],                      # human
    "PRODUCT": ["wd:Q2424752"],               # product
    "QUANTITY": ["wd:Q47574"],                # quantity
    "TIME": ["wd:Q186408"],                   # time
    "WORK_OF_ART": ["wd:Q838948"],            # work of art
}


class Wikifier:
    def __init__(self):
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            return cache
        return {
            "countries": {},
            "entities": {}
        }


    def wikify(self, entities: List[Dict]) -> List[Dict]:
        entities_with_qid = []
        seen = set()

        # Get the dictionary of cached entities
        cached_entities = self.cache.get("entities", {}) 

        for cache_key in cached_entities.keys():
            # Cache keys are in the format 'name|type'.
            try:
                entity_name, entity_type = cache_key.split('|', 1)
                seen.add((entity_name, entity_type))
            except ValueError:
                # Handle case where key might not have the expected format
                logger.warning(f"Skipping malformed cache key: {cache_key}")
                
        logger.debug(f"Cache has {len(seen)} items.")

        for ent in entities:
            entity_name = ent.get("surface")
            entity_type = ent.get("label")

            if not entity_name or not entity_type:
                logger.debug(f"Skipping entity with missing name/type: {ent}")
                continue

            # Skip if already seen
            key = (entity_name.lower(), entity_type.lower())
            if key in seen:
                logger.debug(f"Skipping duplicate/cached entity: {key}")
                continue
            seen.add(key)

            cache_key = f'{entity_name.lower()}|{entity_type.lower()}'
            qid_url = self.cache.get("entities", {}).get(cache_key)
            if qid_url:
                qid = qid_url.split("/")[-1]
                logger.debug(f"Cache hit for {entity_name} ({entity_type}): {qid}")
                
                # Add the result from the cache to the final list and skip SPARQL/network call
                entities_with_qid.append({
                    "surface": entity_name,
                    "label": entity_type,
                    "sparql_safe": sanitize_for_sparql(entity_name),
                    "qid": qid,
                })
                continue
            
            # Not in cache; proceed to query Wikidata
            else:
                safe_name = sanitize_for_sparql(entity_name)
                if not safe_name:
                    logger.debug(f"{entity_name} could not be sanitized")
                    continue

            qid = self.get_qid(safe_name)

            if qid:
                qid_url = f'https://www.wikidata.org/wiki/{qid}'
                self.cache['entities'][cache_key] = qid_url
                logger.debug(f"{entity_name}: {qid}")
            else:
                logger.debug(f"{entity_name}: has no QID")
                self.cache['entities'][cache_key] = None
                qid_url = None

            entities_with_qid.append({
                "surface": entity_name,
                "label": entity_type,
                "sparql_safe": safe_name,
                "qid": qid,
            })

        with open(CACHE_FILE, 'w') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

        return entities_with_qid


    @sleep_and_retry
    @limits(calls=1, period=ONE_SECOND)
    def get_qid(self, entity_name: str, entity_type:str = None) -> str:
        exact_match_qid = self.get_qid_via_exact_match(entity_name, entity_type)
        if exact_match_qid:
            return exact_match_qid

        fuzzy_match_qid = self.query_via_fuzzy_search(entity_name, entity_type)
        if fuzzy_match_qid:
            return fuzzy_match_qid

        return None


    def get_qid_via_exact_match(self, entity_name: str, entity_type: str | None = None) -> str:
        """
        Query Wikidata for an exact English label match and optionally restrict
        results to a specific entity type (e.g., 'country', 'city', 'person').
        """

        # --- Optional type restriction ---
        if entity_type and entity_type in TYPE_QID_MAP:
            qids = " ".join(TYPE_QID_MAP[entity_type])
            type_filter = f"""
                ?item wdt:P31/wdt:P279* ?type .
                VALUES ?type {{ {qids} }} .
            """
        else:
            type_filter = ""

        # --- SPARQL query ---
        query = f"""
            SELECT ?item WHERE {{
            ?item rdfs:label "{entity_name}"@en .
            {type_filter}
            }}
            LIMIT 1
        """

        # --- Execute query ---
        bindings = self.query_via_sparql(query)
        if bindings:
            return bindings[0]["item"]["value"].split("/")[-1]
        return None


    def query_via_sparql(self, query: str):
        """Runs a SPARQL query and returns the parsed JSON response."""
        endpoint_url = "https://query.wikidata.org/sparql"
        headers = {"User-Agent": f"WorldBankKGBot/1.0 ({EMAIL_ADDRESS})"}
        params = {"query": query, "format": "json"}

        response = requests.get(endpoint_url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()

        if "boolean" in results:
            return results["boolean"]
        elif "results" in results and "bindings" in results["results"]:
            return results["results"]["bindings"]
        else:
            raise ValueError(f"Unexpected SPARQL response structure: {results}")

    
    def query_via_fuzzy_search(
            self, 
            entity_name: str, 
            entity_type: str | None = None, 
            limit: int = 10
        ) -> str:
        """
        Type-aware fuzzy search using wbsearchentities, falling back to CirrusSearch
        when necessary.
        """
        headers = {"User-Agent": f"WorldBankKGBot/1.0 ({EMAIL_ADDRESS})"}

        # First, try the standard Wikidata entity search
        params = {
            "action": "wbsearchentities",
            "search": entity_name,
            "language": "en",
            "format": "json",
            "type": "item",
            "limit": limit,
        }
        r = requests.get("https://www.wikidata.org/w/api.php", params=params, headers=headers)
        r.raise_for_status()
        results = r.json().get("search", [])

        # If no results OR all look wrong, fall back to CirrusSearch
        if len(results) <= 1 and entity_type in TYPE_QID_MAP:
            type_qid = TYPE_QID_MAP[entity_type][0].replace("wd:", "")
            logger.debug(f"Falling back to CirrusSearch for {entity_name} ({entity_type})")
            params = {
                "action": "query",
                "list": "search",
                "srsearch": f"{entity_name} haswbstatement:P31={type_qid}",
                "srlimit": limit,
                "format": "json",
            }
            r = requests.get("https://www.wikidata.org/w/api.php", params=params, headers=headers)
            r.raise_for_status()
            cirrus_results = r.json().get("query", {}).get("search", [])
            if cirrus_results:
                # Cirrus returns page titles like 'Q805'; extract first match
                title = cirrus_results[0]["title"]
                if title.startswith("Q"):
                    return title

        # If we have normal results, run batch type check
        if not results:
            return None
        if not entity_type or entity_type not in TYPE_QID_MAP:
            return results[0]["id"]

        candidate_qids = [r["id"] for r in results]
        qid_values = " ".join(f"wd:{qid}" for qid in candidate_qids)
        allowed_types = " ".join(TYPE_QID_MAP[entity_type])
        check_query = f"""
            SELECT ?item WHERE {{
            VALUES ?item {{ {qid_values} }}
            ?item wdt:P31/wdt:P279* ?type .
            VALUES ?type {{ {allowed_types} }} .
            }}
        """
        type_results = self.query_via_sparql(check_query)
        valid_qids = {b["item"]["value"].split("/")[-1] for b in type_results} if type_results else set()

        for res in results:
            if res["id"] in valid_qids:
                return res["id"]

        # Fallback
        return results[0]["id"]


    
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
    

def main():
    entity_linker = Wikifier()
    
    ent = "the World Bank"
    QID = entity_linker.get_qid(ent)

    print(QID)

if __name__ == "__main__":
    main()