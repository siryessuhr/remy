from typing import Any, cast

from langchain_core.tools import tool
from sqlalchemy import Float, literal, select

from remy.database import create_engine, create_session_factory
from remy.models import RecipeModel
from remy.settings import settings
from remy.utils import generate_embeddings


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
    min_score: float = 0.15,
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

    engine = create_engine(settings.DATABASE_URL)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            query_vector = literal(query_embedding)
            ingred_embedding = cast(Any, RecipeModel.ingred_embedding)
            distance_expression = ingred_embedding.op("<=>", return_type=Float)(query_vector)
            statement = (
                select(RecipeModel, distance_expression.label("distance")).order_by(distance_expression).limit(top_k)
            )
            result = await session.execute(statement)
            matches = []
            for recipe, distance in result.all():
                score = _distance_to_score(float(distance))
                if score < min_score:
                    continue
                matches.append(
                    {
                        "id": recipe.id,
                        "url": recipe.url,
                        "title": recipe.title,
                        "ingredients": recipe.ingredients,
                        "labels": recipe.labels,
                        "content": f"{recipe.title}\n{recipe.ingredients}\n{recipe.instructions}",
                        "score": score,
                    }
                )
            return matches
    finally:
        await engine.dispose()


@tool
async def vector_similarity_search_tool(
    query: str,
    top_k: int = 8,
    min_score: float = 0.2,
) -> list[dict[str, Any]]:
    """Search for recipes semantically similar to a natural-language query.

    Args:
        query: User request or recipe description to search for.
        top_k: Maximum number of recipes to return.
        min_score: Minimum similarity score threshold in [0.0, 1.0].

    Returns:
        Ranked recipe matches with metadata and score.
    """
    query_embedding = generate_embeddings(query)
    return await vector_similarity_search(
        query_embedding,
        top_k=top_k,
        min_score=min_score,
    )
