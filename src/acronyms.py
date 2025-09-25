from typing import List, Tuple, Text, Dict
import json
from llama_index.core import Document
from openai import OpenAI
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from pprint import pprint
import html

from src.storage import load_index

class AcronymExtractor:
    def __init__(self):
        self.acronyms = None
        self.entities = None
    
    def _get_acronym_section(self) -> Text:
        index = load_index()

        query_str = """
        Find sections of the document that define acronyms or abbreviations.
        These sections may be called 'Abbreviations', 'Acronyms', or 'List of Acronyms'.
        """
        retriever = index.as_retriever(similarity_top_k=5)

        nodes = retriever.retrieve(query_str)

        for node in nodes:
            print(f"[Score: {node.score:.2f}] {node.node.text[:300]}...\n")

        return "---\n\n".join([node.node.text for node in nodes])
    
    def _extract_acronyms_with_llm(self, text, client=None) -> Dict:
        if not client:
            client = OpenAI()
        
        prompt = """
        Extract a dictionary of acronyms and their definitions from the following text.

        Return as a valid JSON dictionary like: {"ABC": "Definition of ABC", ...}

        Text:
        """ + text

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at understanding document formatting and extracting structured acronym definitions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        declared_acronyms = json.loads(response.choices[0].message.content)

        return declared_acronyms
    
    def _extract_inline_acronyms(self, text) -> Dict:
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("abbreviation_detector")

        doc = nlp(text)

        # Extract abbreviations
        inline_acronyms = {}
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
                    print(f"⚠️ Warning: Conflict for acronym '{abbr}':")
                    print(f"    Primary:  {merged[abbr]}")
                    print(f"    Detected: {definition}")
            else:
                print(f'➕ {abbr}: {definition}')
                merged[abbr] = definition

        return merged
    

    def get_all_entities_from_acronyms(self, primary: dict, detected: dict) -> List:
        # Flip primary
        entities = {v: k for k, v in primary.items()}

        # Flip detected and merge
        for k, v in detected.items():
            entities.setdefault(v, k)

        self.entities = entities

        return self.entities
    

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

        merged_acronyms = self.merge_acronym_dicts(primary_acronyms, secondary_acronyms)

        self.acronyms = merged_acronyms

        # store entities
        entities_from_acronyms = self.get_all_entities_from_acronyms(primary_acronyms, secondary_acronyms)
        
        return self.acronyms

