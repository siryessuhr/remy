# Remy - Recipe Agent

## General

- This project uses `uv` to manage the Python virtual environment and dependencies. All Python commands should be run using `uv run <command>`.
- **ALWAYS** run `poe fmt` before committing code to ensure proper formatting.

## Project Description

Remy is a **Meal and Recipe Agent** built with FastAPI, LangChain/LangGraph, and PostgreSQL (pgvector). It automates the collection, extraction, labeling, and semantic search of recipes from web URLs. Key capabilities:

- **Web Scraping**: Fetches and parses recipe pages using BeautifulSoup4 + lxml, handling auth sites (e.g., NYT Cooking).
- **Recipe Extraction**: Uses an LLM via Ollama to extract structured data (ingredients, instructions) from raw HTML/text.
- **Label Generation**: Auto-generates labels (meal type, cuisine, cooking method) via LangGraph agents and LLM prompting.
- **Vector Search**: Semantic search over recipe metadata using pgvector embeddings.
- **Meal Planning**: Generates personalized meal plans by analyzing recent recipes and ingredient alignment.
- **REST API**: FastAPI endpoints for adding recipes, managing labels, generating meal plans, and querying the database.

The project uses LangSmith for tracing LLM interactions and testing pipeline outputs.

## Tech Stack

**Core Frameworks:**
- **FastAPI** — REST API layer
- **LangChain + LangGraph** — Agent orchestration and workflow graphs (recipe extraction, label generation, meal planning)
- **SQLModel (SQLAlchemy & Pydantic) + Alembic** — ORM and database migrations

**Database:**
- **PostgreSQL** with **pgvector** for semantic embeddings
- Tables: `recipe`, `user_labels`, `meal_plans`

**LLM / AI:**
- **Ollama** (gemma4:31b) for recipe extraction and label generation via LangChain
- **qwen3-embedding:latest** for vector embeddings (512 dimensions)
- **LangSmith** for tracing, datasets, and evaluation

**Web Scraping:**
- **BeautifulSoup4 + lxml** — HTML parsing
- **httpx2** — HTTP client for fetching recipe pages

**Data Validation & Config:**
- **Pydantic v2 + pydantic-settings** — Settings loaded from `.env`
- **loguru** — Structured logging

**Dev Tools:**
- **uv** — Package management and virtual environment
- **Poe the Poet (poe)** — Task runner (`poe fmt`, `poe lint`, `poe test`)
- **ruff** — Linting and formatting
- **pyrefly** — Static type checking
- **pytest + pytest-asyncio + pytest-mock** — Testing

## Key Development Commands

All Python commands should be run using `uv` - prepending all Python commands with `uv run` ensures the correct virtual environment and dependencies.

There are standard commands defined using the `poe` command:
- `poe lint`: Run `ruff` & `pyrefly` type checkers
- `poe fmt`: Format Python code using `ruff`
- `poe test`: Run tests using `pytest`

## Architecture Overview

```
                    ┌─────────────────────┐
                    │      FastAPI         │
                    │   /api/recipes/add   │
                    │   /api/search        │
                    │   /api/meal_plans    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
       ┌─────────────┐  ┌────────────┐  ┌──────────────┐
       │ Recipe Agent│  │Label Agent │  │Meal Plan Agent│
       │ (LangGraph) │  │(LangGraph) │  │ (LangGraph)  │
       └─────────────┘  └────────────┘  └──────────────┘
```

**Core Architecture:**
- **FastAPI REST API**: Exposes endpoints for adding recipes, searching, and generating meal plans
- **LangGraph Agents**: Three main agents handle recipe extraction, label generation, and meal planning
- **PostgreSQL + pgvector**: Stores recipes, embeddings (512-dim), and user labels for semantic search
- **Pylance/Pyright**: Provides type checking and code intelligence

**Data Flow:**
1. Web scraper fetches raw HTML from recipe URLs
2. BeautifulSoup/lxml parses the HTML structure
3. LangChain/LangGraph agent extracts structured data (ingredients, instructions)
4. LLM generates labels (meal type, cuisine, cooking method)
5. Data is upserted into PostgreSQL with pgvector embeddings
6. Semantic search queries use pgvector for vector similarity matching

**Key Components:**
- `src/remy/app.py` — FastAPI application entry point
- `src/remy/settings.py` — Pydantic settings (DB, Ollama, LangSmith config)
- `src/remy/agents/` — LangGraph agents for recipe extraction and label generation
- `src/remy/tools/` — Reusable tools (url_fetcher, ingredient_extractor, vector_search, etc.)
- `src/remy/database/` — Asyncpg connection pool, SQLAlchemy models, Alembic migrations
- `tests/` — Unit and integration tests

**Environment Variables:**
Required in `.env`:
```
# Database Config
DATABASE_URL=postgresql://postgres:password@localhost:5432/database_name

# Ollama Config
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:31b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
OLLAMA_EMBEDDING_DIMENSIONS=512

# FastAPI Config
APP_HOST=0.0.0.0
APP_PORT=8000
DEV=true

# LangSmith Config
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<API KEY>
LANGSMITH_PROJECT="LangSmith Project Name"
```
