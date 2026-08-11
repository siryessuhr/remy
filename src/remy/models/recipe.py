from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from pydantic import ConfigDict
from sqlmodel import Field, SQLModel


class BaseRecipeModel(SQLModel):
    """Base recipe model for shared attributes."""

    url: str
    title: str
    ingredients: str
    instructions: str


class RecipeModel(BaseRecipeModel, table=True):
    """Recipe record with pgvector embedding support."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int | None = Field(default=None, primary_key=True)
    labels: str = "[]"
    # pyrefly: ignore [no-matching-overload]
    ingred_embedding: VECTOR = Field(default=None, sa_type=VECTOR(512))
    # pyrefly: ignore [no-matching-overload]
    instru_embedding: VECTOR = Field(default=None, sa_type=VECTOR(512))
    score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
