import networkx as nx
from typing import List, Dict, Optional
from collections import defaultdict
import logging
from dotenv import load_dotenv
from collections import Counter
from openai import OpenAI
import tiktoken
import json
from pathlib import Path
import requests

from graspologic.partition import hierarchical_leiden
from rdflib import URIRef, Literal, RDF
import matplotlib.pyplot as plt

from src.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="secrets/.env")


class OllamaClient:
    """
    Minimal drop-in replacement for OpenAI chat.completions.create,
    using the local Ollama REST API.
    """
    def __init__(self, base_url="http://localhost:11434/api/generate", model="llama3"):
        self.base_url = base_url
        self.model = model

    def chat(self, messages, temperature=0.3):
        # Combine all messages into a single prompt
        prompt = "\n".join([m["content"] for m in messages])

        data = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False
        }

        logger.debug(f"Sending prompt to Ollama model '{self.model}'...")
        resp = requests.post(self.base_url, json=data)
        resp.raise_for_status()
        response = resp.json()

        # Return an object mimicking OpenAI’s response
        return {"choices": [{"message": {"content": response["response"]}}]}
    

class Summarizer:
    def __init__(self, kg: KnowledgeGraph, client=None, backend="openai", cache_path="summaries_cache.json", max_context_tokens=8000):
        self.kg = kg
        self.client = client
        self.backend = backend
        self.max_context_tokens = max_context_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # TODO
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()
        if not client:
            self.client = OpenAI()

    
    def _load_cache(self):
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text."""
        return len(self.encoding.encode(text))
    

    def _call_summary_model(self, text_block: str) -> str:
        """Summarize using either OpenAI API or Ollama."""
        prompt = f"""
        Summarize the following group of text passages from World Bank documents.
        Identify the main topics, entities, and development themes.
        Use a concise and factual tone suitable for knowledge graph metadata.

        Text:
        {text_block[:10000]}
        """

        messages = [
            {"role": "system", "content": "You are an expert in summarizing development project documents."},
            {"role": "user", "content": prompt}
        ]

        if self.backend == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        elif self.backend == "ollama":
            response = self.client.chat(messages, temperature=0.3)
            return response["choices"][0]["message"]["content"].strip()

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        
    def build_chunk_graph(self) -> nx.Graph:
        """
        Build an undirected weighted graph where:
        - Nodes are chunk URIs
        - Edges connect chunks sharing at least one entity
        - Edge weight = number of shared entities
        """
        G = nx.Graph()
        chunk_to_entities = defaultdict(set)
        entity_to_chunks = defaultdict(set)

        for chunk_uri, _, entity_uri in self.kg.g.triples((None, self.kg.schema.mentions, None)):
            if "chunk/" in str(chunk_uri):
                chunk_to_entities[chunk_uri].add(entity_uri)
                entity_to_chunks[entity_uri].add(chunk_uri)

        for entity, chunks in entity_to_chunks.items():
            chunk_list = list(chunks)
            for i in range(len(chunk_list)):
                for j in range(i + 1, len(chunk_list)):
                    u, v = chunk_list[i], chunk_list[j]
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += 1
                    else:
                        G.add_edge(u, v, weight=1)

        logger.info(f"Graph built with {G.number_of_nodes()} chunks and {G.number_of_edges()} edges")
        return G


    def detect_communities_hierarchical_leiden(self, G: nx.Graph):
        """
        Run hierarchical Leiden community detection on the chunk graph.
        Returns:
            community_mapping: dict(chunk_uri → community_id)
            hierarchy: dict of levels (optional)
        """
        if G.number_of_edges() == 0:
            logger.warning("Graph has no edges — skipping community detection")
            return {}, None

        hc = hierarchical_leiden(
            G,
            weight_attribute="weight",
            max_cluster_size=50,
            resolution=1.25,  # Controls granularity (i.e., number of communities)
            random_seed=42
        )

        chunk_to_comm = hc.first_level_hierarchical_clustering()
        counts = Counter(chunk_to_comm.values())
        logger.info(f"Detected {len(counts)} communities at top level")
        logger.info(f"Top-level community sizes: {counts.most_common(10)}")

        return chunk_to_comm, hc


    def add_communities_to_graph(self, chunk_to_comm: dict):
        graph = self.kg.g

        for chunk_uri_str, comm_id in chunk_to_comm.items():
            chunk_uri = URIRef(chunk_uri_str)
            comm_uri = self.kg.ex[f"community/{comm_id}"]
            graph.add((chunk_uri, self.kg.schema.isPartOf, comm_uri))
        
            # Add community node type & label (if not already present)
            if not list(graph.triples((comm_uri, RDF.type, None))):
                graph.add((comm_uri, RDF.type, self.kg.schema.Community))
                graph.add((comm_uri, self.kg.schema.identifier, Literal(str(comm_id))))
                graph.add((comm_uri, self.kg.schema.name, Literal(f"Community {comm_id}")))

            # Link chunk to community
            graph.add((chunk_uri, self.kg.schema.isPartOf, comm_uri))

        self.kg.save()

        logger.info(f"Added {len(chunk_to_comm)} chunk→community links to graph")

    
    def summarize_communities(self, chunk_to_comm: dict, max_tokens_per_summary=6000):
        """
        Summarize each community of chunks, respecting the token limit.
        Returns: dict(comm_id → summary_text)
        """
        logger.info("Grouping chunks by community for summarization...")
        comm_to_texts = defaultdict(list)

        for chunk_uri, comm_id in chunk_to_comm.items():
            for _, _, text in self.kg.g.triples((chunk_uri, self.kg.schema.text, None)):
                comm_to_texts[comm_id].append(str(text))

        summaries = self.cache.copy()

        logger.info(f"Resuming from cache: {len(self.cache)} summaries already done.")
        logger.info(f"{len(comm_to_texts) - len(self.cache)} communities remaining.")

        for comm_id, texts in comm_to_texts.items():
            combined = ""
            for t in texts:
                if self._estimate_tokens(combined + t) < max_tokens_per_summary:
                    combined += "\n" + t
                else:
                    break  # stop adding once near limit

            if not combined.strip():
                logger.warning(f"Community {comm_id} has no text — skipping.")
                continue

            logger.info(f"Summarizing community {comm_id} with ~{self._estimate_tokens(combined)} tokens...")
            
            summary = self._call_summary_model(combined)
            if summary:
                self.cache[str(comm_id)] = summary
                self._save_cache()

                # Store in KG
                comm_uri = self.kg.ex[f"community/{comm_id}"]
                self.kg.g.add((comm_uri, self.kg.schema.abstract, Literal(summary)))
            else:
                logger.debug(f"No summary returned for community {comm_id}")

        self._save_cache()
        self.kg.save()

        logger.info(f"Completed {len(summaries)} total summaries.")
        
        return summaries


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    client = OllamaClient(model="llama3.2:latest")
    kg = KnowledgeGraph.load_or_build('world-bank-kg.ttl', rebuild=False)
    summarizer = Summarizer(kg, client, backend='ollama')
    chunk_graph = summarizer.build_chunk_graph()
    chunk_to_comm, hc = summarizer.detect_communities_hierarchical_leiden(chunk_graph)
    summarizer.add_communities_to_graph(chunk_to_comm)
    summarizer.summarize_communities(chunk_to_comm)

    summarizer.kg.save()

    
    # Show graph
    # plt.figure(figsize=(10, 8))
    # nx.draw(
    #     chunk_graph,
    #     node_size=10,
    #     edge_color="lightgray",
    #     with_labels=False
    # )

    # plt.title("Chunk-Entity Graph")
    # plt.savefig("output/chunk_graph.png", dpi=300, bbox_inches="tight")



if __name__ == '__main__':
    main()