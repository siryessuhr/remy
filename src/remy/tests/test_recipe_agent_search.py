"""Tests for RecipeAgent semantic search behavior."""

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import Float, literal

from remy.agents.recipe_agent import RecipeAgent
from remy.models import RecipeExtractionState, RecipeModel
from remy.tools.vector_search import vector_similarity_search


def test_vector_distance_expression_returns_float_distance():
    """Distance queries must produce numeric values, not pgvector objects."""
    expression = RecipeModel.ingred_embedding.op("<=>", return_type=Float)(literal([0.1, 0.2, 0.3]))

    assert isinstance(expression.type, Float)


@pytest.mark.asyncio
async def test_stream_search_intent_emits_semantic_results(mocker):
    """Search intent should emit vector-search progress and result payload."""

    agent = RecipeAgent(llm=mocker.MagicMock())

    mocker.patch.object(
        agent,
        "_understand_user_intent",
        return_value={"user_intent": "search_recipe_in_db", "url": ""},
    )
    mocker.patch.object(
        agent,
        "_respond_with_search_results",
        return_value={
            "search_results": [
                {
                    "title": "Tomato Basil Pasta",
                    "url": "https://example.com/pasta",
                    "score": 0.91,
                }
            ],
            "search_response": {
                "message": "I found one strong match.",
                "recommendations": [
                    {
                        "title": "Tomato Basil Pasta",
                        "url": "https://example.com/pasta",
                        "score": 0.91,
                        "reason": "Closest ingredient profile.",
                    }
                ],
            },
        },
    )

    # pyrefly: ignore [bad-argument-type]
    events = [event async for event in agent._stream_with_session("simple pasta", session=object())]

    assert events[0] == {"type": "progress", "message": "Understanding your request..."}
    assert events[1] == {"type": "progress", "message": "Searching and preparing recommendations..."}
    assert events[2]["type"] == "result"
    # pyrefly: ignore [bad-index]
    assert events[2]["payload"]["matched_recipes"][0]["title"] == "Tomato Basil Pasta"
    # pyrefly: ignore [bad-index]
    assert events[2]["payload"]["response"]["message"] == "I found one strong match."


@pytest.mark.asyncio
async def test_respond_with_search_results_uses_langchain_tool_and_summarizes(mocker):
    """Search response node should call the tool and summarize returned matches."""

    llm = mocker.MagicMock()
    tool_enabled_llm = mocker.MagicMock()
    tool_enabled_llm.ainvoke = mocker.AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "vector_similarity_search_tool",
                        "args": {"query": "quick chicken dinner", "top_k": 8, "min_score": 0.35},
                        "id": "tool-call-1",
                    }
                ],
            ),
            # pyrefly: ignore [no-matching-overload]
            AIMessage(
                content=(
                    '{"message":"Top match found.","recommendations":[{"title":"Roast Chicken",'
                    '"url":"https://example.com/chicken","score":0.84,"reason":"Closest to the requested meal."}]}'
                )
            ),
        ]
    )
    llm.bind_tools.return_value = tool_enabled_llm

    agent = RecipeAgent(llm=llm)
    state = RecipeExtractionState(user_request="quick chicken dinner")

    tool_mock = mocker.MagicMock()
    tool_mock.ainvoke = mocker.AsyncMock(
        return_value=[
            {
                "title": "Roast Chicken",
                "url": "https://example.com/chicken",
                "score": 0.84,
            }
        ]
    )
    mocker.patch("remy.agents.recipe_agent.vector_similarity_search_tool", tool_mock)

    result = await agent._respond_with_search_results(state)

    llm.bind_tools.assert_called_once()
    tool_mock.ainvoke.assert_awaited_once_with({"query": "quick chicken dinner", "top_k": 8, "min_score": 0.35})
    assert result["search_results"] == [
        {
            "title": "Roast Chicken",
            "url": "https://example.com/chicken",
            "score": 0.84,
        }
    ]
    # pyrefly: ignore [bad-index]
    assert result["search_response"]["message"] == "Top match found."


@pytest.mark.asyncio
async def test_vector_similarity_search_uses_existing_recipe_schema(mocker):
    """Search should use the app's RecipeModel schema instead of a langchain-only vector table."""
    recipe = mocker.Mock(
        id=7,
        url="https://example.com/pasta",
        title="Pasta",
        ingredients="pasta, olive oil",
        labels="dinner",
        instructions="Cook pasta",
    )
    recipe.ingred_embedding = [0.1, 0.2, 0.3]

    session = mocker.AsyncMock()
    session.execute = mocker.AsyncMock(return_value=mocker.Mock(all=lambda: [(recipe, 0.25)]))

    session_cm = mocker.MagicMock()
    session_cm.__aenter__ = mocker.AsyncMock(return_value=session)
    session_cm.__aexit__ = mocker.AsyncMock(return_value=None)

    engine = mocker.Mock()
    engine.dispose = mocker.AsyncMock()
    mocker.patch("remy.tools.vector_search.create_engine", return_value=engine)
    mocker.patch("remy.tools.vector_search.create_session_factory", return_value=lambda: session_cm)

    results = await vector_similarity_search([0.1, 0.2, 0.3], top_k=5, min_score=0.0)

    assert results[0]["id"] == 7
    assert results[0]["title"] == "Pasta"
    assert results[0]["score"] > 0.0
