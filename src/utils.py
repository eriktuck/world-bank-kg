import re
import tiktoken

def sanitize_for_sparql(entity: str) -> str | None:
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


def num_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
