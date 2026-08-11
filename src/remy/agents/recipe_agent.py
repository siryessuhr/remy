"""Remy the Recipe Agent."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Self

import httpx2
from bs4 import BeautifulSoup
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger as log
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from remy.agents.prompts import (
    GENERATE_LABELS_PROMPT,
    RECIPE_EXTRACTION_PROMPT,
    SEARCH_RECIPES_RESPONSE_PROMPT,
    USER_INTENT_PROMPT,
)
from remy.database import create_engine, create_session_factory
from remy.models import BaseRecipeModel, RecipeExtractionState, RecipeModel
from remy.settings import settings
from remy.tools.vector_search import vector_similarity_search_tool
from remy.utils import generate_embeddings, get_llm, parse_structured_response, with_structured_output


class RecipeAgent:
    """Agent that extracts a recipe from a given text input."""

    def __init__(
        self,
        llm,
        session: AsyncSession | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        """Initialize the recipe agent and optional async session factory.

        Args:
            llm: LLM instance used by graph nodes.
            session: Optional active async session used by DB nodes.
            session_factory: Optional async session factory for persistence.
        """
        self.llm = llm
        self.session = session
        self.session_factory = session_factory

    @classmethod
    def from_env(cls) -> Self:
        """Build a recipe agent from environment settings.

        Returns:
            Configured recipe agent.
        """
        engine = create_engine(settings.DATABASE_URL)
        session_factory = create_session_factory(engine)

        return cls(
            llm=get_llm(),
            session_factory=session_factory,
        )

    def _create_graph(self, state_schema: type[BaseModel]) -> CompiledStateGraph:
        """Create the recipe extraction LangGraph."""
        # Need to create the following:
        # 1. start node that evaluates user input
        # 1a. If the user input is a URL, fetch the recipe from the URL and extract it
        # 1b. If the user input is text, evaluate if its a recipe and extract the recipe from the text
        # 1c. If the user input is a request for a recipe, search the database for a recipe and return it
        # 1d. If the user input is a request for a meal plan, evaluate the content and generate a proposed
        #     meal plan with recipes from the database based on similarity of the request. If the user doesn't
        #     provide enough information, ask the user for more information about their target incredients,
        #     dietary restrictions, and preferences. Then generate a meal plan with recipes from the database.

        # 1a. Nodes: fetch_url -> parse_html -> extract_recipe
        #     -> generate_embeddings -> generate_labels -> persist to db
        # 1b. Nodes: extract_recipe -> generate_embeddings -> generate_labels -> persist to db
        # 1c. Nodes: search_db_with_vectors -> return_recipe
        # 1d. Nodes: generate_meal_plan -> generate_embeddings -> generate_labels -> persist to db

        # End is when either content is persisted to db or a recipe is returned to the user. The graph should be able to
        # handle multiple paths and return the appropriate response based on the user input.
        graph = StateGraph(state_schema)

        graph.add_node("understand_user_intent", self._understand_user_intent)
        graph.add_node("fetch_recipe_from_url", self._fetch_recipe_from_url)
        graph.add_node("parse_html", self._parse_html)
        graph.add_node("extract_recipe", self._extract_recipe)
        graph.add_node("generate_labels", self._generate_labels)
        graph.add_node("generate_embeddings", self._generate_embeddings)
        graph.add_node("insert_to_db", self._insert_to_db)
        graph.add_node("respond_with_search_results", self._respond_with_search_results)

        graph.set_entry_point("understand_user_intent")

        graph.add_edge("fetch_recipe_from_url", "parse_html")
        graph.add_edge("parse_html", "extract_recipe")
        graph.add_edge("extract_recipe", "generate_labels")
        graph.add_edge("generate_labels", "generate_embeddings")
        graph.add_edge("generate_embeddings", "insert_to_db")
        graph.add_edge("insert_to_db", END)
        graph.add_edge("respond_with_search_results", END)
        graph.add_conditional_edges(
            "understand_user_intent",
            lambda state: state.user_intent.value if state.user_intent is not None else "search_recipe_in_db",
            {
                "extract_recipe_from_url": "fetch_recipe_from_url",
                "extract_recipe_from_text": "extract_recipe",
                "search_recipe_in_db": "respond_with_search_results",
                "generate_meal_plan": "respond_with_search_results",
            },
        )

        return graph.compile()

    def extract_recipe(self, text: str) -> BaseRecipeModel:
        """Extract a recipe from plain text using a synchronous LLM call.

        Args:
            text: Raw user-provided recipe text.

        Returns:
            Parsed recipe model.
        """
        prompt = ChatPromptTemplate.from_template(RECIPE_EXTRACTION_PROMPT)
        chain = prompt | self.llm
        log.info("Extracting recipe from text input (sync).")
        response = chain.invoke({"text": text})
        log.debug(f"Raw LLM response: {response}")
        return _parse_recipe_model_from_response(response)

    async def _understand_user_intent(self, state: RecipeExtractionState) -> dict[str, str | None]:
        """Understand the user's intent from their request.

        Args:
            state: Current recipe extraction state.

        Returns:
            User intent payload used for graph routing.
        """
        prompt = ChatPromptTemplate.from_template(USER_INTENT_PROMPT)
        structured_llm = with_structured_output(self.llm, dict[str, str])
        chain = prompt | structured_llm
        log.info("Understanding user intent...")
        response = await chain.ainvoke({"user_request": state.user_request})
        log.debug(f"Raw LLM response: {response}")
        if isinstance(response, dict):
            result = response
        else:
            result = parse_structured_response(response, dict[str, str])
        return {
            "user_intent": result["user_intent"],
            "url": result.get("url", ""),
        }

    async def _fetch_recipe_from_url(self, state: RecipeExtractionState) -> dict[str, str | None]:
        """Fetch the recipe text from the given URL.

        Args:
            state: Current recipe extraction state.

        Returns:
            HTML payload fetched from the target URL.
        """
        log.info(f"Fetching recipe from URL: {state.url}")
        async with httpx2.AsyncClient(timeout=30.0) as client:
            resp = await client.get(state.url)
            resp.raise_for_status()

            return {"raw_html": resp.text}

    async def _parse_html(self, state: RecipeExtractionState) -> dict[str, str | None]:
        """Parse the HTML content and extract the text.

        Args:
            state: Current recipe extraction state.

        Returns:
            Parsed text body extracted from HTML.
        """
        log.info("Parsing HTML content...")
        soup = BeautifulSoup(state.raw_html, "lxml")
        parsed_body = soup.get_text(separator="\n", strip=True)
        return {"parsed_body": parsed_body}

    async def _extract_recipe(self, state: RecipeExtractionState) -> dict[str, str | None]:
        """Extract a recipe from text using the LLM.

        Args:
            state: Current recipe extraction state.

        Returns:
            Processed recipe payload validated against BaseRecipeModel.
        """
        prompt = ChatPromptTemplate.from_template(RECIPE_EXTRACTION_PROMPT)
        structured_llm = with_structured_output(self.llm, BaseRecipeModel)
        chain = prompt | structured_llm
        log.info("Extracting recipe from text input...")
        source_text = state.parsed_body or state.user_request
        response = await chain.ainvoke({"text": source_text})
        log.debug(f"Raw LLM response: {response}")
        if isinstance(response, BaseRecipeModel):
            recipe = response
        else:
            recipe = _parse_recipe_model_from_response(response)
        log.info(f"Extracted recipe: {recipe}")
        # pyrefly: ignore [bad-assignment]
        return {"processed_recipe": recipe}

    async def _generate_labels(self, state: RecipeExtractionState) -> dict[str, str | None]:
        """Generate labels for the extracted recipe.

        Args:
            state: Current recipe extraction state.

        Returns:
            Label list payload extracted from the model response.
        """
        log.info("Generating labels for the extracted recipe...")
        prompt = ChatPromptTemplate.from_template(GENERATE_LABELS_PROMPT)
        structured_llm = with_structured_output(self.llm, dict[str, list[str]])
        chain = prompt | structured_llm
        response = await chain.ainvoke(
            {
                # pyrefly: ignore [missing-attribute]
                "ingredients": state.processed_recipe.ingredients,
                # pyrefly: ignore [missing-attribute]
                "instructions": state.processed_recipe.instructions,
            }
        )
        log.debug(f"Raw LLM response: {response}")

        if isinstance(response, list):
            label_payload = {"labels": response}
        elif isinstance(response, dict):
            label_payload = response
        else:
            label_payload = parse_structured_response(response, dict[str, list[str]])
            if isinstance(label_payload, list):
                label_payload = {"labels": label_payload}
        if not isinstance(label_payload, dict) or "labels" not in label_payload:
            message = f"Label generation did not return a valid labels payload: {label_payload!r}"
            raise ValueError(message)

        # pyrefly: ignore [missing-attribute]
        return {"labels": label_payload["labels"]}

    async def _generate_embeddings(self, state: RecipeExtractionState) -> dict[str, RecipeModel]:
        """Generate embeddings for both ingredients and instructions.

        Args:
            state: Current recipe extraction state.

        Returns:
            Recipe model enriched with vector embeddings.
        """
        log.info("Generate embeddings for recipe ingredients & instructions.")
        # pyrefly: ignore [missing-attribute]
        ing_embed = await asyncio.to_thread(generate_embeddings, state.processed_recipe.ingredients)
        # pyrefly: ignore [missing-attribute]
        inst_embed = await asyncio.to_thread(generate_embeddings, state.processed_recipe.instructions)

        recipe = RecipeModel(
            ingred_embedding=ing_embed,
            instru_embedding=inst_embed,
            labels=f"{state.labels}",
            # pyrefly: ignore [missing-attribute]
            **state.processed_recipe.model_dump(),
        )
        log.debug(f"Final recipe model: {recipe}")

        return {"processed_recipe": recipe}

    async def _insert_to_db(self, state: RecipeExtractionState) -> dict[str, RecipeModel]:
        """Insert a processed recipe in the database.

        Args:
            state: Current recipe extraction state.

        Returns:
            Updated state payload containing the persisted recipe.

        Raises:
            ValueError: If there is no valid processed recipe to persist.
        """
        recipe = state.processed_recipe
        if not isinstance(recipe, RecipeModel):
            message = "Cannot persist recipe: processed_recipe must be a RecipeModel."
            raise ValueError(message)

        if self.session is None:
            message = "Cannot persist recipe: no active database session is available."
            raise ValueError(message)

        self.session.add(recipe)
        await self.session.commit()
        await self.session.refresh(recipe)

        log.info(f"Inserted recipe into database: {recipe.title} (ID: {recipe.id})")
        return {"processed_recipe": recipe}

    async def _respond_with_search_results(self, state: RecipeExtractionState) -> dict[str, object]:
        """Generate guarded recommendations with LLM tool-calling.

        Args:
            state: Current recipe extraction state.

        Returns:
            Payload containing retrieved matches and a summarized recommendation.
        """
        prompt = ChatPromptTemplate.from_template(SEARCH_RECIPES_RESPONSE_PROMPT)
        prompt_messages = prompt.format_messages(user_request=state.user_request)
        llm_with_tools = self.llm.bind_tools([vector_similarity_search_tool])
        initial_response = await llm_with_tools.ainvoke(prompt_messages)

        if not isinstance(initial_response, AIMessage):
            fallback = _parse_search_response(str(getattr(initial_response, "content", "")))
            return {"search_results": [], "search_response": fallback}

        tool_calls = list(initial_response.tool_calls or [])
        if not tool_calls:
            parsed_response = _parse_search_response(str(initial_response.content))
            return {"search_results": [], "search_response": parsed_response}

        first_tool_call = tool_calls[0]
        tool_args = first_tool_call.get("args", {})
        tool_output = await vector_similarity_search_tool.ainvoke(tool_args)
        matches = _normalize_search_matches(tool_output)
        log.info("Tool retrieved {} semantically similar recipes.", len(matches))

        tool_message = ToolMessage(
            content=json.dumps(matches),
            tool_call_id=str(first_tool_call.get("id", "vector-search-call")),
        )
        final_response = await llm_with_tools.ainvoke(
            [
                *prompt_messages,
                initial_response,
                tool_message,
            ]
        )

        final_content = str(getattr(final_response, "content", ""))
        parsed_response = _parse_search_response(final_content)
        return {"search_results": matches, "search_response": parsed_response}

    async def stream(self, user_query: str, session: AsyncSession | None = None) -> AsyncIterator[dict[str, object]]:
        """Stream recipe-agent progress updates while processing a user query."""
        if session is not None:
            scoped_agent = RecipeAgent(
                llm=self.llm,
                session=session,
                session_factory=self.session_factory,
            )
            async for event in scoped_agent._stream_with_session(user_query, session):
                yield event
            return

        if self.session is not None:
            async for event in self._stream_with_session(user_query, self.session):
                yield event
            return

        if self.session_factory is None:
            message = "No database session available. Provide a session or initialize with from_env()."
            raise ValueError(message)

        async with self.session_factory() as internal_session:
            scoped_agent = RecipeAgent(
                llm=self.llm,
                session=internal_session,
                session_factory=self.session_factory,
            )
            async for event in scoped_agent._stream_with_session(user_query, internal_session):
                yield event

    async def _stream_with_session(
        self,
        user_query: str,
        session: AsyncSession | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        """Process the recipe pipeline and emit progress updates for a given session."""
        yield {"type": "progress", "message": "Understanding your request..."}
        state = RecipeExtractionState(user_request=user_query)

        intent_state = await self._understand_user_intent(state)
        state = state.model_copy(update=intent_state)

        if state.user_intent == "search_recipe_in_db":
            yield {"type": "progress", "message": "Searching and preparing recommendations..."}
            state = state.model_copy(update=await self._respond_with_search_results(state))
            yield {
                "type": "result",
                "payload": {
                    "user_request": user_query,
                    "matched_recipes": state.search_results,
                    "response": state.search_response,
                },
            }
            return

        if state.user_intent == "extract_recipe_from_url":
            yield {"type": "progress", "message": "Fetching the recipe from the provided URL..."}
            state = state.model_copy(update=await self._fetch_recipe_from_url(state))
            yield {"type": "progress", "message": "Parsing the recipe content..."}
            state = state.model_copy(update=await self._parse_html(state))
        else:
            yield {"type": "progress", "message": "Extracting the recipe from your request..."}

        yield {"type": "progress", "message": "Extracting the recipe details..."}
        state = state.model_copy(update=await self._extract_recipe(state))
        yield {"type": "progress", "message": "Generating labels..."}
        state = state.model_copy(update=await self._generate_labels(state))
        yield {"type": "progress", "message": "Generating embeddings..."}
        state = state.model_copy(update=await self._generate_embeddings(state))

        if session is None:
            message = "A database session is required to persist the extracted recipe."
            raise ValueError(message)

        state = state.model_copy(update=await self._insert_to_db(state))
        yield {
            "type": "result",
            "payload": {
                "user_request": user_query,
                "recipe": state.processed_recipe.model_dump() if state.processed_recipe else None,
            },
        }

    async def invoke(self, user_query: str, session: AsyncSession | None = None) -> dict[str, object]:
        """Invoke the recipe extraction agent with a user query."""
        if session is not None:
            scoped_agent = RecipeAgent(
                llm=self.llm,
                session=session,
                session_factory=self.session_factory,
            )
            return await scoped_agent.invoke(user_query)

        state = RecipeExtractionState(user_request=user_query)

        if self.session is not None:
            state_graph = self._create_graph(RecipeExtractionState)
            return await state_graph.ainvoke(state)

        if self.session_factory is None:
            message = "No database session available. Provide a session or initialize with from_env()."
            raise ValueError(message)

        async with self.session_factory() as internal_session:
            scoped_agent = RecipeAgent(
                llm=self.llm,
                session=internal_session,
                session_factory=self.session_factory,
            )
            return await scoped_agent.invoke(user_query)


def _parse_search_response(raw_content: str) -> dict[str, object]:
    """Parse search-response JSON from the LLM with a defensive fallback.

    Args:
        raw_content: Raw string emitted by the LLM.

    Returns:
        Parsed response dictionary with message and recommendations.
    """
    try:
        parsed = json.loads(raw_content.strip())
    except json.JSONDecodeError:
        return {"message": raw_content.strip(), "recommendations": []}

    message = parsed.get("message")
    recommendations = parsed.get("recommendations")
    if not isinstance(message, str):
        message = "Here are the closest matching recipes I found."
    if not isinstance(recommendations, list):
        recommendations = []

    return {"message": message, "recommendations": recommendations}


def _normalize_search_matches(tool_output: Any) -> list[dict[str, Any]]:
    """Normalize vector-search tool output to a list of dictionaries.

    Args:
        tool_output: Raw payload returned by the LangChain tool.

    Returns:
        List of dictionary search results.
    """
    if not isinstance(tool_output, list):
        return []

    return [item for item in tool_output if isinstance(item, dict)]


def _parse_recipe_model_from_response(response: object) -> BaseRecipeModel:
    """Parse a recipe model from an LLM response object.

    Args:
        response: LLM response that is expected to include a ``content`` attribute.

    Returns:
        Parsed base recipe model.
    """
    raw_content = str(getattr(response, "content", "")).strip()
    return BaseRecipeModel.model_validate_json(raw_content)
