"""LangGraph state models for Remy the Recipe Agent."""

from enum import StrEnum

from pydantic import BaseModel


class UserIntent(StrEnum):
    """User intent for the recipe extraction sub-graph."""

    EXTRACT_RECIPE_FROM_URL = "extract_recipe_from_url"
    EXTRACT_RECIPE_FROM_TEXT = "extract_recipe_from_text"
    SEARCH_RECIPE_IN_DB = "search_recipe_in_db"
    GENERATE_MEAL_PLAN = "generate_meal_plan"


class RecipeExtractionState(BaseModel):
    """LangGraph state for the recipe extraction sub-graph."""

    user_request: str

    user_intent: UserIntent | None = None

    url: str = ""
    raw_html: str = ""
    parsed_body: str = ""
    extracted: dict = {}  # {"ingredients": [...], "instructions": [...]}
    labels: list[str] = []
    extraction_valid: bool = False
