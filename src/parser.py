import json
import re
from typing import List, Tuple, Optional
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship
import logging
logger = logging.getLogger(__name__)


class CustomParser:
    def __init__(self, include_metadata=False, include_prev_next_rel=False, chunk_overlap: int = 0):
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel
        self.chunk_overlap = chunk_overlap

    # TODO: integrate LlamaIndex SentenceSplitter or SemanticSplitter
    def _split_long_text(self, text: str, max_chars: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current = [], []

        for sent in sentences:
            if not sent:
                continue
            candidate = " ".join(current + [sent])
            if len(candidate) <= max_chars:
                current.append(sent)
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [sent]

        if current:
            chunks.append(" ".join(current))

        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev_tail = chunks[i-1][-self.chunk_overlap:]
                    overlapped.append(prev_tail + " " + chunk)
                else:
                    overlapped.append(chunk)
            return overlapped
        return chunks
    
    
    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Ensure metadata only has flat values (str, int, float, None)."""
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)) or v is None:
                clean[k] = v
            else:
                # Fallback: stringify complex values
                clean[k] = json.dumps(v, ensure_ascii=False)
        return clean


    def get_nodes_from_documents(self, docs: List[Document], max_chars: int = 1000) -> List[TextNode]:
        nodes: List[TextNode] = []

        for doc in docs:
            try:
                struct_out = json.loads(doc.text)
            except json.JSONDecodeError:
                logger.warning(f"Skipping ill-formed text: {doc.text[:50]}")
                continue

            header_stack: List[Tuple[int, str]] = []

            buffer: List[str] = []
            buffer_metadata: List[dict] = []
            buffer_len = 0
            buffer_headers: Optional[List[Tuple[int, str]]] = None

            def merge_page_indices(metadata_list: List[dict]):
                """Collect page_idx values into a list if multiple."""
                pages = []
                for md in metadata_list:
                    if "page_idx" in md:
                        if isinstance(md["page_idx"], list):
                            pages.extend(md["page_idx"])
                        else:
                            pages.append(md["page_idx"])
                # Deduplicate + preserve order
                seen = set()
                return [p for p in pages if not (p in seen or seen.add(p))]

            def flush_buffer():
                nonlocal buffer, buffer_metadata, buffer_len, buffer_headers
                if not buffer:
                    return
                combined_text = "\n\n".join(buffer)
                metadata = doc.metadata.copy()
                if self.include_metadata:
                    for md in buffer_metadata:
                        metadata.update(md)
                    # collect page indices into list
                    pages = merge_page_indices(buffer_metadata)
                    if pages:
                        metadata["page_idx"] = ",".join(str(p) for p in pages)
                # Use the headers that were active when this buffer was built
                metadata["headers"] = " > ".join([h for _, h in (buffer_headers or header_stack)])
                nodes.append(
                    TextNode(
                        text=combined_text,
                        ref_doc_id=doc.doc_id,
                        metadata=self._sanitize_metadata(metadata),
                    )
                )
                buffer, buffer_metadata, buffer_len, buffer_headers = [], [], 0, None

            for element in struct_out:
                etype = element.get("type")

                # ---- Handle headers ----
                if etype == "text" and "text_level" in element and element["text_level"]:
                    flush_buffer()
                    level = element["text_level"]
                    text = element.get("text", "").strip()
                    if not text:
                        continue
                    if level == 1:
                        header_stack = [(1, text)]
                    else:
                        header_stack = [(lvl, h) for lvl, h in header_stack if lvl < level]
                        header_stack.append((level, text))
                    continue

                # ---- Handle text content ----
                if etype == "text":
                    content = element.get("text", "").strip()
                    if not content:
                        continue

                    # --- Header-consistency guard ---
                    if buffer and buffer_headers is not None and buffer_headers != header_stack:
                        flush_buffer()
                    if buffer_headers is None:
                        buffer_headers = header_stack.copy()
                    # --------------------------------

                    # Split if content itself is longer than max_chars
                    if len(content) > max_chars:
                        flush_buffer()
                        chunks = self._split_long_text(content, max_chars)
                        for chunk in chunks:
                            metadata = doc.metadata.copy()
                            if self.include_metadata:
                                for k, v in element.items():
                                    if k not in {"type", "text"}:
                                        metadata[k] = v
                            metadata["headers"] = " > ".join([h for _, h in (buffer_headers or header_stack)])
                            nodes.append(
                                TextNode(
                                    text=chunk,
                                    ref_doc_id=doc.doc_id,
                                    metadata=self._sanitize_metadata(metadata),
                                )
                            )
                        continue

                    md = {}
                    if self.include_metadata:
                        md = {k: v for k, v in element.items() if k not in {"type", "text"}}
                    if buffer_len + len(content) > max_chars:
                        flush_buffer()
                    buffer.append(content)
                    buffer_metadata.append(md)
                    buffer_len += len(content)
                    continue

                # ---- Handle tables/images ----
                if etype in ("table", "image"):
                    flush_buffer()
                    metadata = doc.metadata.copy()
                    if self.include_metadata:
                        for k, v in element.items():
                            if k not in {"type", "table_body", "img_path"}:
                                metadata[k] = v
                    metadata["headers"] = " > ".join([h for _, h in (buffer_headers or header_stack)])
                    if etype == "table":
                        text_value = element.get("table_body", "").strip()
                        if not text_value:
                            logger.warning("Skipping empty table element")
                            continue
                    elif etype == "image":
                        text_value = element.get("img_path", "").strip()
                        if not text_value:
                            logger.warning("Skipping empty image element")
                            continue
                    nodes.append(
                        TextNode(
                            text=text_value,
                            ref_doc_id=doc.doc_id,
                            metadata=self._sanitize_metadata(metadata),
                        )
                    )
                    continue

                # ---- Unknown element type ----
                logger.warning(f"Skipping unknown element type: {etype}")
                flush_buffer()
                continue

            flush_buffer()

        # ---- Add prev/next if requested ----
        if self.include_prev_next_rel:
            for i, node in enumerate(nodes):
                if i > 0:
                    node.relationships[NodeRelationship.PREVIOUS] = nodes[i - 1].as_related_node_info()
                if i < len(nodes) - 1:
                    node.relationships[NodeRelationship.NEXT] = nodes[i + 1].as_related_node_info()

        return nodes
