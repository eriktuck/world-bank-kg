import json
import re
from typing import List, Tuple, Optional, Sequence, Any
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import Document
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import TextNode, NodeRelationship, BaseNode
from llama_index.core.utils import get_tqdm_iterable
from bs4 import BeautifulSoup
from markdownify import markdownify as md
                
import logging

logger = logging.getLogger(__name__)


class CustomParser(NodeParser):
    """
    Custom parser for MinerU structured output.

    Splits a sequence of documents into Nodes, converting
    headers to metadata and chunking long text segments.

    Headers trigger new sections. Text under headers grows
    until the next header is encountered; long text segments 
    are chunked using the specified text splitter.

    Tables are handled separately and are chunked as needed.
    Text around a table is kept together until the next header.

    Images are currently skipped.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
        header_path_separator (str): separator char used for section header path metadata
        chunk_size (int): size of text chunks
        chunk_overlap (int): overlap size between text chunks
    """

    header_path_separator: str = Field(
        default="/", description="Separator char used for section header path metadata."
    )

    text_splitter: SentenceSplitter = Field(
        default_factory=SentenceSplitter, 
        description="The text splitter to use for long sections."
    )

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        header_path_separator: str = "/",
        callback_manager: Optional[CallbackManager] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
    ) -> "CustomParser":
        callback_manager = callback_manager or CallbackManager([])
        
        text_splitter = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            header_path_separator=header_path_separator,
            callback_manager=callback_manager,
            text_splitter=text_splitter,
        )
    
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)
        
        return all_nodes


    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from a single BaseNode."""
        try:
            struct_out = json.loads(node.text)
        except json.JSONDecodeError:
            logger.warning(f"Skipping ill-formed text: {node.text[:50]}")
            return []
        
        text_nodes: List[TextNode] = []
        current_section = ""
        header_stack: List[Tuple[int, str]] = []

        def process_current_section(current_section: str):
            if not current_section.strip():
               return ""
            
            header_path = self.header_path_separator.join(
                [h for _, h in header_stack]
            )

            text_splits = self.text_splitter.split_text(current_section.strip())
            new_nodes = self._build_nodes_from_splits(
                text_splits,
                node,
                header_path
            )
            text_nodes.extend(new_nodes)
            
            # reset buffer
            return ""

        for element in struct_out:
            etype = element.get("type")

            # ---- Handle headers ----
            if (etype == "text" and "text_level" in element 
                and isinstance(element["text_level"], int)):
                # Save the previous section before starting a new one
                current_section = process_current_section(current_section)
                
                level = element["text_level"]
                text = element.get("text", "").strip()
                if not text:
                    continue
                if level == 1:
                    header_stack = [(1, text)]
                else:
                    header_stack = [(lvl, h) for lvl, h in header_stack if lvl < level]
                    header_stack.append((level, text))
                current_section = "#" * level + f" {text}\n"
                continue
            
            # ---- Handle text elements ----
            elif etype == "text":
                current_section += element.get("text", "") + "\n"
            
            # ---- Handle table elements ----
            elif etype == "table":
                table_body = element.get("table_body", "")
                table_soup = BeautifulSoup(table_body, 'html.parser')
                table_element = table_soup.find('table')
                html_str = str(table_element)
                markdown_output = md(
                    html_str, 
                    # Use a specific style for headings if needed, e.g., 'ATX'
                    heading_style="ATX" 
                )

                process_current_section(markdown_output)

            # ---- Handle image elements ----
            elif etype == "image":
                print("Image encountered; skipping for now.")
            
            else:
                logger.warning(f"Unknown element type: {etype}")
        
        # Save any remaining section
        process_current_section(current_section)
        
        return text_nodes
    

    def _build_nodes_from_splits(
        self,
        text_splits: List[str],
        node: BaseNode,
        header_path: str,
    ) -> List[TextNode]:
        """Build nodes from a list of text splits."""
        # Use LlamaIndex's utility to create nodes and handle prev/next relationships
        nodes = build_nodes_from_splits(
            text_splits, 
            node, 
            id_func=self.id_func,
        )
        
        if self.include_metadata:
            separator = self.header_path_separator
            header_path = (
                separator + header_path + separator if header_path else separator
            )
            for n in nodes:
                # Add the header path metadata to every chunk
                n.metadata["header_path"] = header_path

        return nodes


def main():
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG)

    fixture_path = Path("tests") / "fixtures" / "structured_sample.json"
    with open(fixture_path, 'r') as f:
        struct_out = f.read()

    doc = Document(
        text=struct_out,
        metadata={"source": str(fixture_path)}
    )

    parser = CustomParser.from_defaults(
        include_metadata=True, 
        include_prev_next_rel=True,
        chunk_size=100,
        chunk_overlap=40)
    nodes = parser.get_nodes_from_documents([doc])

    for i, node in enumerate(nodes):
        print(f"--- Node {i} ---")
        print(f"Text: {node.text}")
        print(f"Metadata: {node.metadata}")
        print(f"Relationships: {node.relationships}")
        print()
    print(f"Total nodes created: {len(nodes)}")


if __name__ == "__main__":
    main()
    