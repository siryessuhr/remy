# Recipe Agent Remy

The idea behind the recipe agent was largely selfish - my wife and I enjoy cooking together so how could I potentially leverage an LLM powered application to save recipes to a database (build an app to just view those recipes later), then query recipes using natural language while providing either types of food or ingredients that we had on-hand. Some recipes call for a little cabbage, for instance, but the grocery store typically sells bags that are more than one meal. Having an agent do a little lifting using something like a RAG tool to find multiple recipes that have a common ingredient could be helpful.

This could be more broadly useful to a chef at a restaurant with their own custom recipes and the same predicament - local or seasonal ingredients that you want to ensure all get used to reduce waste.

## Local Setup (no Docker)

If you, like me, have a computer capable of self-hosting LLMs and already have Ollama running, and you don't mind the overhead of setting up and managing a PostgreSQL database...

### Tools required

- `uv` Python environment & dependency manager for backend
- `npm` Javascript dependency manager for frontend
- PostgreSQL
- Either OpenAI access or a local Ollama install

## LLM provider selection

Remy supports both OpenAI and Ollama. The app chooses the provider automatically:

- Default behavior: use OpenAI
- Override behavior: if `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are both set, use Ollama instead
- The same logic is used for both chat completion and embeddings

This means you can keep one default setup in `.env` without changing code, and switch to a local Ollama stack simply by filling in the Ollama variables.

### Environment variables

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/recipe_agent

# OpenAI (default provider)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536

# Optional Ollama override
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=qwen3.6:latest
# OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
# OLLAMA_EMBEDDING_DIMENSIONS=512
```

### Which provider is used?

The selection rule is intentionally simple:

```python
if OLLAMA_BASE_URL and OLLAMA_MODEL:
    use Ollama
else:
    use OpenAI
```

This keeps the app predictable while still allowing local/self-hosted inference when desired.

> Quick note on embeddings: the model used to generate recipe vectors and the model used to embed search queries must match. If one side uses OpenAI embeddings and the other uses Ollama embeddings, the vectors will not be comparable and similarity search will be unreliable.

## Database Pre-Setup

Before running migrations or starting the API, ensure PostgreSQL and pgvector are ready.

### Requirements

- PostgreSQL 16+ (or a Postgres image with pgvector installed, such as `pgvector/pgvector:pg16`)
- A database matching `DATABASE_URL` in `.env`
- A DB user with permission to create extensions (or have an admin pre-create the extension)

### One-time database setup

1. Create the database if it does not exist.
2. Enable pgvector in that database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If your DB user cannot create extensions, ask an admin to run that statement once.

### Generate and apply migrations

Use the project CLI wrappers:

```bash
uv run python -m remy migrate revision --message "Initial schema" --autogenerate
uv run python -m remy migrate upgrade head
```

### Startup behavior

The startup script runs migrations before launching the API. On a fresh setup with no migration files, it bootstraps the recipe table. Once revision files exist, it runs normal Alembic upgrades.

### Local Ollama example

If you want to use a local Ollama instance instead of OpenAI, uncomment and set:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.6:latest
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
OLLAMA_EMBEDDING_DIMENSIONS=512
```

When these are provided, Remy automatically switches over to Ollama for both chat generation and embedding generation.

## Local Setup with Docker

### Tools required

- Docker

### Docker quickstart (local)

Use this flow for first-time local setup with Docker Compose:

```bash
docker compose up -d
```

Then open the frontend at http://localhost:5173.

Notes:

- `app` startup waits for Postgres readiness and runs migrations automatically.
- The configured Postgres image (`pgvector/pgvector:pg16`) includes pgvector, and the initial migration creates the `vector` extension.
- The frontend service exposes the Vite dev server on port 5173 and proxies API calls to the backend container.

## Top-Level Architecture

Single entry point with intent-based routing to 3 parallel paths. Each path handles a specific user intent (URL extraction, text extraction, or recipe search) and converges to an appropriate final state.

### Unified Graph

```mermaid
graph TD
    user["User input / request"] --> router{Intent Router<br/>node}

    router -- "URL" --> pathA[Path A: URL Extraction]
    router -- "Recipe text" --> pathB[Path B: Text Extraction]
    router -- "Search recipe" --> pathC[Path C: Recipe Search]

    %% Path A — URL extraction
    subgraph "A. URL Extraction"
        fetchUrl["fetch_url<br/>(HTTP client)"] --> parseHtml["parse_html<br/>→ extract HTML text"]
        parseHtml --> extractRecipe1["extract_recipe<br/>→ LLM extracts recipe"]
        extractRecipe1 --> generateEmbeddings1["generate_embeddings"]
        generateEmbeddings1 --> generateLabels1["generate_labels"]
        generateLabels1 --> persistA["persist to db<br/>(database_upsert)"]
    end

    %% Path B — Text extraction
    subgraph "B. Text Extraction"
        extractRecipe2["extract_recipe<br/>→ LLM extracts recipe"]
        extractRecipe2 --> generateEmbeddings2["generate_embeddings"]
        generateEmbeddings2 --> generateLabels2["generate_labels"]
        generateLabels2 --> persistB["persist to db<br/>(database_upsert)"]
    end

    pathA --> fetchUrl
    pathB --> extractRecipe2

    %% Path C — Recipe search
    subgraph "C. Recipe Search"
        searchDb["search_db_with_vectors<br/>(pgvector cosine distance)"] --> returnRecipe["return_recipe<br/>(formatted response)"]
    end

    pathC --> searchDb

    %% Final nodes
    persistA --> persistEnd1["✅ Persisted to DB"]
    persistB --> persistEnd2["✅ Persisted to DB"]
    returnRecipe --> searchEnd["✅ Recipe returned<br/>to user"]

    classDef startNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef routerNode fill:#fff9c4,stroke:#f9a825,stroke-width:2px;
    classDef toolNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef llmNode fill:#fce4ec,stroke:#ad1457,stroke-width:1px,stroke-dasharray:5 5;
    classDef decision fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef finalNode fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px;

    class user startNode;
    class router decision;
    class pathA,pathB,pathC llmNode;
    class fetchUrl,parseHtml,generateEmbeddings1,generateLabels1,persistA,generateEmbeddings2,generateLabels2,persistB,searchDb,returnRecipe toolNode;
    class extractRecipe1,extractRecipe2 llmNode;
    class persistEnd1,persistEnd2,searchEnd finalNode;
```

---

## Path A — URL Extraction Detail

**Trigger:** User submits a URL to a recipe page.
**Flow:** Fetch HTML → parse text → extract via LLM → generate embeddings & labels → persist.

```mermaid
graph TD
    router{Intent: URL?} -- yes --> fetchUrl["fetch_url<br/>(HTTP GET)"]
    router -- no --> other["→ other path"]

    fetchUrl --> html["HTML response"]
    html --> parseHtml["parse_html<br/>→ clean &<br/>extract text content"]
    parseHtml --> recipeText["plain recipe text"]

    recipeText --> extractRecipe["extract_recipe<br/>(LLM call)<br/>→ BaseRecipeModel"]
    extractRecipe --> embeddings["generate_embeddings<br/>(recipe vector)"]
    embeddings --> labels["generate_labels<br/>(cuisine, difficulty,<br/>tags, nutrition)"]

    labels --> persist["persist to db<br/>(database_upsert)"]
    persist --> saved["✅ Recipe saved"]

    classDef routerNode fill:#fff9c4,stroke:#f9a825,stroke-width:2px;
    classDef toolNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef llmNode fill:#fce4ec,stroke:#ad1457,stroke-width:1px,stroke-dasharray:5 5;
    classDef decision fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef finalNode fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px;

    class router decision;
    class extractRecipe,llmNode llmNode;
    class fetchUrl,parseHtml,persist toolNode;
    class saved finalNode;
```

---

## Path B — Text Extraction Detail

**Trigger:** User pastes recipe text (no URL needed).
**Flow:** Direct LLM extraction → embeddings & labels → persist.

```mermaid
graph TD
    router{Intent: Recipe<br/>text?} -- yes --> extractRecipe["extract_recipe<br/>(LLM call)<br/>→ BaseRecipeModel"]
    router -- no --> other["→ other path"]

    extractRecipe --> embeddings["generate_embeddings<br/>(recipe vector)"]
    embeddings --> labels["generate_labels<br/>(cuisine, difficulty,<br/>tags, nutrition)"]

    labels --> persist["persist to db<br/>(database_upsert)"]
    persist --> saved["✅ Recipe saved"]

    classDef routerNode fill:#fff9c4,stroke:#f9a825,stroke-width:2px;
    classDef toolNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef llmNode fill:#fce4ec,stroke:#ad1457,stroke-width:1px,stroke-dasharray:5 5;
    classDef decision fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef finalNode fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px;

    class router,hasResults decision;
    class parseQuery,llmNodeA llmNode;
    class searchDb toolNode;
    class returned,returned2 finalNode;
```

---

## Path C — Recipe Search Detail

**Trigger:** User submits a natural-language search query.
**Flow:** Parse query → vector similarity search → return ranked results.

```mermaid
graph TD
    router{Intent: Search<br/>recipe?} -- yes --> parseQuery["parse_query<br/>(LLM extracts filters)"]
    router -- no --> other["→ other path"]

    parseQuery --> searchDb["search_db_with_vectors<br/>(pgvector cosine distance)"]
    searchDb --> hasResults{results found?}
    hasResults -- yes --> returned["return_recipe<br/>(formatted response)"]
    hasResults -- no fallback --> returned2["empty / partial-match response"]

    returned --> logSave["search_logger<br/>(via LLM)"]
    logSave --> logSaved["✅ Logged"]
    returned2 --> logSave2["search_logger<br/>(via LLM)"]
    logSave2 --> logSaved2["✅ Logged"]

    classDef routerNode fill:#fff9c4,stroke:#f9a825,stroke-width:2px;
    classDef toolNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef llmNode fill:#fce4ec,stroke:#ad1457,stroke-width:1px,stroke-dasharray:5 5;
    classDef decision fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef finalNode fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px;

    class router,hasResults decision;
    class parseQuery,llmNodeA llmNode;
    class searchDb toolNode;
    class returned,returned2 finalNode;
```

---

## Design Notes

- I have been interested in self-hosting apps at home for many reasons, so leveraging Ollama was a priority - unfortunately I have some performance issues I need to resolve with my Ollama configuration. I decided to add OpenAI support after some slow development cycles and to better support execution by others.

- The frontend is extremely simple and basically forwards all information to the user - this is because the current audience is technical.

- I set it up so that the only database write interactions are static and not provided as a tool to an LLM. The vector similarity search as a LangChain tool seems like a reasonable tradeoff for security purposes on the database.

- All agent nodes & edges are defined in a single class but this could absolutely be designed as an agent of agents.

## What I would improve given more time

- I don't love the agent class - I think there's some opportunity for clean-up, refinement, refactor, etc, especially around readability.

- Refactor the frontend to be less logger verbose (which I kept for demonstration purposes), and return better natural language responses.

- Add a "manual" recipe look up - don't make it exclusively via LLM/agent.

- My first intuitive step is to write Python as a backend application then forward that to a frontend in a different language. Since LangChain supports Javascript/Typescript, I think it would be interesting to approach this entirely in Javascript/Typescript instead.

- Some models across OpenAI and Ollama support varying levels of structured outputs and even tool calling. Since both are critical to this workflow, I would add more guardrails around configuration.

- I spent a fair amount of time on little things that improve or manage extensibility - Alembic for database management, uv with Docker, some tests. I would improve all of these with more time.
