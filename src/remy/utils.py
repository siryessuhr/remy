import json
from typing import Any, TypeVar, get_origin

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger as log

from remy.settings import settings

T = TypeVar("T")


def use_ollama() -> bool:
    """Return True when the app should use Ollama instead of the default OpenAI provider.

    Returns:
        Whether Ollama has been explicitly configured via environment variables.
    """
    return bool(settings.OLLAMA_BASE_URL and settings.OLLAMA_MODEL)


def get_llm(*, format: str | None = "json"):
    """Create a chat LLM using the configured provider.

    Args:
        format: JSON output format for Ollama-compatible clients.

    Returns:
        A configured chat model instance for either OpenAI or Ollama.
    """
    if use_ollama():
        log.info("Using Ollama LLM provider: {}", settings.OLLAMA_MODEL)
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            format=format or "json",
            temperature=0,
        )

    if settings.OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY is required when Ollama is not configured.")

    log.info("Using OpenAI LLM provider: {}", settings.OPENAI_MODEL)
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )


def get_embeddings_client():
    """Create the configured embeddings client.

    Returns:
        A configured embedding model instance for the active provider.
    """
    if use_ollama():
        log.info("Using Ollama embeddings provider: {}", settings.OLLAMA_EMBEDDING_MODEL)
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBEDDING_MODEL,
            dimensions=settings.OLLAMA_EMBEDDING_DIMENSIONS,
        )

    if settings.OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY is required when Ollama is not configured.")

    log.info("Using OpenAI embeddings provider: {}", settings.OPENAI_EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        dimensions=settings.OPENAI_EMBEDDING_DIMENSIONS,
    )


def with_structured_output(model: Any, schema: type[T]) -> Any:
    """Return a model wrapper that supports structured output across providers.

    OpenAI models support ``with_structured_output`` directly. Ollama models do not,
    so we fall back to the regular model and parse the JSON content ourselves.

    Args:
        model: LangChain chat model instance.
        schema: Pydantic schema or dict-like structure to coerce into.

    Returns:
        A model wrapper or a compatible fallback that returns parsed objects.
    """
    if hasattr(model, "with_structured_output"):
        try:
            return model.with_structured_output(schema)
        except TypeError:
            log.warning("Structured output requested but unavailable for model; falling back to JSON parsing.")

    return model


def parse_structured_response(response: Any, schema: type[T] | dict[str, Any]) -> T:
    """Parse JSON content from a model response into the requested schema.

    Args:
        response: LLM response object containing content.
        schema: Expected schema type or dict definition.

    Returns:
        Parsed structured response.
    """
    raw_content = str(getattr(response, "content", "")).strip()
    payload = json.loads(raw_content)
    if schema is dict or get_origin(schema) is dict:
        return payload
    if isinstance(schema, type) and hasattr(schema, "model_validate"):
        return schema.model_validate(payload)
    return payload


def generate_embeddings(text: str) -> list[float]:
    """Generate an embedding vector for text using the active provider.

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
