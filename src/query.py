import logging
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

from src.storage import load_existing_index  # same loader used in your pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

Settings.llm = Ollama(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.2,
    additional_kwargs={"num_ctx": 2048}
)
Settings.embed_model = Settings.embed_model


def start_chat():
    """Interactive chat over the unified World Bank KG vector index."""
    logger.info("Loading existing vector index...")
    index: VectorStoreIndex = load_existing_index()

    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=Settings.llm,
        response_mode="compact",
    )

    # Create a retriever-backed conversational engine
    memory = ChatMemoryBuffer.from_defaults(token_limit=6000)
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        llm=Settings.llm,
        memory=memory,
        context_prompt=(
            "You are an expert analyst for World Bank environmental project data. "
            "Use the provided documents and community summaries to give factual, "
            "concise answers. When helpful, include numeric values or project IDs."
        ),
    )

    print("\nWorld Bank Knowledge Graph Chat")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            query = input("User: ").strip()
            if query.lower() in {"exit", "quit"}:
                print("\nExiting chat.")
                break

            response = chat_engine.chat(query)

            print("\nResponse:", response.response.strip(), "\n")

            # Optional: show short source snippets
            if hasattr(response, "source_nodes"):
                print("Sources:")
                for node in response.source_nodes[:3]:
                    meta = node.metadata or {}
                    snippet = node.text[:140].replace("\n", " ")
                    print(f" - [{meta.get('type', 'chunk')}] {snippet}...")
                print()

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    start_chat()
