"""Remy the Recipe Agent."""

import json

import httpx2
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger as log
from pydantic import BaseModel

from remy.agents.prompts import RECIPE_EXTRACTION_PROMPT, USER_INTENT_PROMPT
from remy.models import BaseRecipeModel, RecipeExtractionState
from remy.settings import settings


class RecipeAgent:
    """Agent that extracts a recipe from a given text input."""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    @classmethod
    def from_env(cls):
        return cls(
            llm=ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                format="json",
            ),
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

        # 1a. Nodes: fetch_url -> parse_html -> extract_recipe -> generate_embeddings -> generate_labels -> persist to db
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

        graph.set_entry_point("understand_user_intent")

        graph.add_edge("fetch_recipe_from_url", "parse_html")
        graph.add_edge("parse_html", "extract_recipe")
        graph.add_conditional_edges(
            "understand_user_intent",
            lambda state: state.user_intent,
            {
                "extract_recipe_from_url": "fetch_recipe_from_url",
                "extract_recipe_from_text": "extract_recipe",
            },
        )

        return graph.compile()

    def _understand_user_intent(self, state: RecipeExtractionState) -> dict[str, str | None]:
        """Understand the user's intent from their request."""
        prompt = ChatPromptTemplate.from_template(USER_INTENT_PROMPT)
        chain = prompt | self.llm
        log.info("Understanding user intent...")
        response = chain.invoke({"user_request": state.user_request})
        log.debug(f"Raw LLM response: {response}")
        result = json.loads(response.content.strip())
        return {
            "user_intent": result["user_intent"],
            "url": result.get("url"),
        }

    def _fetch_recipe_from_url(self, state: RecipeExtractionState) -> str:
        """Fetch the recipe text from the given URL."""
        log.info(f"Fetching recipe from URL: {state.url}")
        with httpx2.Client(timeout=30.0) as client:
            resp = client.get(state.url)
            resp.raise_for_status()

            return {"raw_html": resp.text,}

    def _parse_html(self, state: RecipeExtractionState) -> str:
        """Parse the HTML content and extract the text."""
        soup = BeautifulSoup(state.raw_html, "lxml")
        parsed_body = soup.get_text(separator="\n", strip=True)
        return {"parsed_body": parsed_body}

    def _extract_recipe(self, state: RecipeExtractionState) -> BaseRecipeModel:
        """Extract a recipe from the given text input using the LLM."""
        prompt = ChatPromptTemplate.from_template(RECIPE_EXTRACTION_PROMPT)
        chain = prompt | self.llm
        log.info("Extracting recipe from text input...")
        response = chain.invoke({"text": state.user_request})
        log.debug(f"Raw LLM response: {response}")
        return BaseRecipeModel.model_validate_json(response.content.strip())

    def invoke(self, user_query: str) -> BaseRecipeModel:
        """Invoke the recipe extraction agent with a user query."""
        state = RecipeExtractionState(user_request=user_query)
        state_graph = self._create_graph(RecipeExtractionState)
        return state_graph.invoke(state)
