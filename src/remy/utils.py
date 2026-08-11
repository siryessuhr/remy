from langchain_ollama import OllamaEmbeddings
from loguru import logger as log

from remy.settings import settings


def get_embeddings_client() -> OllamaEmbeddings:
    """Create an Ollama embeddings client.

    Returns:
        Configured ``OllamaEmbeddings`` instance.
    """
    return OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_EMBEDDING_MODEL,
        dimensions=settings.OLLAMA_EMBEDDING_DIMENSIONS,
    )


def generate_embeddings(text: str) -> list[float]:
    """Generate an embedding vector for text using OllamaEmbeddings.

    Args:
        text: Source text to embed.

    Returns:
        Embedding vector.
    """
    log.info("Generating embeddings for text input...")
    embeddings_client = get_embeddings_client()
    embedding = embeddings_client.embed_query(text)
    log.debug("Generated embedding vector with dimension: {}", len(embedding))
    return embedding
