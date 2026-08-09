from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlmodel import Field, SQLModel


class BaseRecipeModel(SQLModel):
    """Base recipe model for shared attributes."""

    url: str
    title: str
    ingredients: str
    instructions: str


class RecipeModel(BaseRecipeModel, table=True):
    """Recipe record with pgvector embedding support."""

    id: int | None = Field(default=None, primary_key=True)
    labels: str = "[]"
    ingred_embedding: VECTOR = Field(default=None, sa_type=VECTOR(768))
    instru_embedding: VECTOR = Field(default=None, sa_type=VECTOR(768))
    score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True
