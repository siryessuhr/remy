from typing import Any

from langchain_postgres import PGEngine, PGVectorStore

from remy.settings import settings
from remy.utils import get_embeddings_client


def _distance_to_score(distance: float) -> float:
    """Convert vector distance to similarity score.

    Args:
        distance: Distance returned by PGVectorStore (lower is better).

    Returns:
        Similarity score where higher is better, clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, 1.0 - distance))


def _document_to_result(document: Any, score: float) -> dict[str, Any]:
    """Convert a LangChain document into the public search result shape.

    Args:
        document: Document returned by PGVectorStore similarity search.
        score: Normalized similarity score.

    Returns:
        A serializable dictionary representing the search hit.
    """
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    return {
        "id": metadata.get("id"),
        "url": metadata.get("url", ""),
        "title": metadata.get("title", ""),
        "ingredients": metadata.get("ingredients", ""),
        "labels": metadata.get("labels", ""),
        "content": document.page_content,
        "score": score,
    }


async def vector_similarity_search(
    query_embedding: list[float],
    *,
    table_name: str = "recipes",
    embedding_column: str = "embedding",
    top_k: int = 10,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Perform vector similarity search using langchain_postgres PGVectorStore.

    Args:
        query_embedding: Precomputed query embedding vector.
        table_name: Vector table name used by PGVectorStore.
        embedding_column: Vector column name used by PGVectorStore.
        top_k: Maximum number of nearest neighbors to return.
        min_score: Minimum similarity score threshold in [0.0, 1.0].

    Returns:
        Ranked, filtered search hits.
    """
    if not query_embedding:
        return []

    engine = PGEngine.from_connection_string(settings.DATABASE_URL)
    try:
        vector_store = PGVectorStore.create(
            engine=engine,
            embedding_service=get_embeddings_client(),
            table_name=table_name,
            embedding_column=embedding_column,
        )

        matches = await vector_store.asimilarity_search_with_score_by_vector(
            embedding=query_embedding,
            k=top_k,
        )
        return [
            _document_to_result(document=document, score=_distance_to_score(distance))
            for document, distance in matches
            if _distance_to_score(distance) >= min_score
        ]
    finally:
        engine.close()


async def vector_similarity_search_tool(
    query_embedding: list[float],
    table_name: str = "recipes",
    embedding_column: str = "embedding",
    top_k: int = 10,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Tool wrapper for vector similarity search."""
    return await vector_similarity_search(
        query_embedding,
        table_name=table_name,
        embedding_column=embedding_column,
        top_k=top_k,
        min_score=min_score,
    )
