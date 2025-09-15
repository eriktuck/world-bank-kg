# src/models/custom_parser.py

import json
from typing import List
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship


class CustomParser:
    def __init__(self, include_metadata: bool = False, include_prev_next_rel: bool = False):
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel

    def get_nodes_from_documents(self, docs: List[Document]) -> List[TextNode]:
        nodes: List[TextNode] = []

        for doc in docs:
            try:
                struct_out = json.loads(doc.text)
            except json.JSONDecodeError:
                continue

            for element in struct_out:
                metadata = doc.metadata.copy()

                # Merge all element-level fields into metadata (except the main text payloads)
                if self.include_metadata:
                    for key, value in element.items():
                        if key not in {"text", "table_body", "img_path"}:
                            metadata[key] = value

                # Determine text content
                node_text = ""
                if element["type"] == "text":
                    node_text = element.get("text", "")
                elif element["type"] == "table":
                    node_text = element.get("table_body", "")
                elif element["type"] == "image":
                    node_text = element.get("img_path", "")
                else:
                    node_text = f"[Unknown element type: {element.get('type')}]"

                node = TextNode(text=node_text, metadata=metadata)
                nodes.append(node)

        if self.include_prev_next_rel:
            for i, node in enumerate(nodes):
                if i > 0:
                    node.relationships[NodeRelationship.PREVIOUS] = nodes[i - 1].as_related_node_info()
                if i < len(nodes) - 1:
                    node.relationships[NodeRelationship.NEXT] = nodes[i + 1].as_related_node_info()
                    
        return nodes
