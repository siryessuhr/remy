from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from loguru import logger as log

from remy.agents.prompts import RECIPE_EXTRACTION_PROMPT
from remy.models.recipe import BaseRecipeModel
from remy.settings import settings


class RecipeAgent:
    """Agent that extracts a recipe from a given text input."""

    def __init__(self, llm):
        self.llm = llm

    @classmethod
    def from_env(cls):
        return cls(
            llm=ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                format="json",
            )
        )

    def extract_recipe(self, text: str) -> BaseRecipeModel:
        """Extract a recipe from the given text input using the LLM."""
        prompt = ChatPromptTemplate.from_template(RECIPE_EXTRACTION_PROMPT)
        chain = prompt | self.llm
        log.info("Extracting recipe from text input...")
        response = chain.invoke({"text": text})
        log.info(f"Raw LLM response: {response}")
        return BaseRecipeModel.model_validate_json(response.content.strip())
