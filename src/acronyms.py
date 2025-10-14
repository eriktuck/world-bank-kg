from typing import List, Tuple, Text, Dict
import json
from openai import OpenAI
import html
import logging
import re

from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

from src.storage import load_index

logger = logging.getLogger(__name__)

class AcronymExtractor:
    def __init__(self, doc_id: str, client=None, backend="openai"):
        self.doc_id = doc_id
        self.client = client
        self.backend = backend
        if not client:
            self.client = OpenAI()
        
        self.acronyms = None
        self.entities = None
    

    def _get_acronym_section(self) -> Text:
        index = load_index()

        query_str = """
        Find sections of the document that define acronyms or abbreviations.
        These sections may be called 'Abbreviations', 'Acronyms', or 'List of Acronyms'.
        """
        retriever = index.as_retriever(
            similarity_top_k=5,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="ref_doc_id", value=str(self.doc_id))]
            ),
        )

        nodes = retriever.retrieve(query_str)

        if not nodes:
            logger.warning(f"No nodes retrieved for doc_id={self.doc_id}")

        results = []
        for scored in nodes:
            node = scored.node
            logger.debug(
                f"[Score: {scored.score:.2f}] "
                f"ref_doc_id={getattr(node, 'ref_doc_id', None)} "
                f"metadata={node.metadata} "
                f"text={node.text[:200]}..."
            )
            results.append(node.text)

        return "---\n\n".join(results)
    

    def _extract_acronyms_with_llm(self, text) -> Dict:

        
        prompt = """
        Extract a dictionary of acronyms and their definitions from the following text.

        Return as a valid JSON dictionary like: {"ABC": "Definition of ABC", ...}

        Text:
        """ + text[:10000]

        logger.debug(f"Prompt sent for acronyms \n\n {prompt}")

        messages=[
            {"role": "system", "content": "You are an expert at understanding document formatting and extracting structured acronym definitions. Always output strict JSON only."},
            {"role": "user", "content": prompt}
        ]

        if self.backend == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
            )
            response_content = response.choices[0].message.content

        elif self.backend == "ollama":
            response = self.client.chat(messages, temperature=0.0)
            response_content = response["choices"][0]["message"]["content"]
        
        logger.debug(f'Model returns result {response_content}')

        # Defensive parsing
        try:
            return json.loads(response_content)

        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse extracted JSON block — trying to repair it.")
                    json_str = re.sub(r"(?<=\w)'(?=\w)", '"', json_str)
                    json_str = re.sub(r",\s*}", "}", json_str)
                    return json.loads(json_str)
            else:
                logger.error("No JSON object found in model output.")
                return {}
    

    def _extract_inline_acronyms(self, doc) -> Dict:
        inline_acronyms = {}
        if not hasattr(doc._, "abbreviations"):
            logger.warning("AbbreviationDetector not in pipeline; skipping acronym extraction.")
            return inline_acronyms
        
        for abrv in doc._.abbreviations:
            inline_acronyms[abrv.text] = abrv._.long_form.text

        inline_acronyms = self.clean_acronyms(inline_acronyms)
        
        return inline_acronyms
    

    def merge_acronym_dicts(self, primary: dict, detected: dict) -> Dict:
        """Merge two acronym dictionaries with a warning on conflicting definitions.

        Args:
            primary (dict): Existing acronym glossary (e.g., from acronym section)
            detected (dict): Acronyms detected from full text using SciSpacy

        Returns:
            dict: Merged dictionary with priority to `primary`
        """
        merged = primary.copy()
        
        for abbr, definition in detected.items():
            if abbr in merged:
                if merged[abbr] != definition:
                    logger.debug(f"⚠️ Warning: Conflict for acronym '{abbr}':")
                    logger.debug(f"    Primary:  {merged[abbr]}")
                    logger.debug(f"    Detected: {definition}")
            else:
                logger.debug(f'➕ {abbr}: {definition}')
                merged[abbr] = definition

        return merged
    

    def get_all_entities_from_acronyms(self, primary: dict, detected: dict) -> List:
        # Flip primary
        entities = {v: k for k, v in primary.items()}

        # Flip detected and merge
        for k, v in detected.items():
            entities.setdefault(v, k)

        return entities
    

    def clean_acronyms(self, acronym_dict: dict, min_upper_ratio: float = 0.5) -> dict:
        """Clean an acronym dictionary by decoding HTML entities in definitions 
        and filtering acronyms that are not sufficiently uppercase.

        Args:
            acronym_dict (dict): Dictionary of acronyms and their definitions.
            min_upper_ratio (float): Minimum ratio of uppercase letters required to keep the acronym.

        Returns:
            dict: Cleaned acronym dictionary.
        """
        cleaned = {}

        for abbr, defn in acronym_dict.items():
            
            if not abbr or 11 > len(abbr) < 2:
                continue
            
            # Remove acronyms that don't meet the uppercase threshold
            num_upper = sum(1 for c in abbr if c.isupper())
            ratio_upper = num_upper / len(abbr)

            if ratio_upper < min_upper_ratio:
                continue

            # Decode HTML entities in definition
            cleaned_defn = html.unescape(defn).strip()
            cleaned[abbr] = cleaned_defn

        return cleaned

    
    def extract(self, text: str) -> Dict:
        # extract and merge acronyms
        acronym_section = self._get_acronym_section()
        primary_acronyms = self._extract_acronyms_with_llm(acronym_section)
        secondary_acronyms = self._extract_inline_acronyms(text)

        # get and store acronyms
        self.acronyms = self.merge_acronym_dicts(primary_acronyms, secondary_acronyms)

        # get and store entities from acronyms
        self.entities = self.get_all_entities_from_acronyms(primary_acronyms, secondary_acronyms)
        
        return self.acronyms

