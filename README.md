# Recipe Agent LangGraph Diagrams

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

### Docker quickstart (local)

Use this flow for first-time local setup with Docker Compose:

```bash
docker compose up -d postgres
docker compose ps postgres
docker compose up -d app
```

Notes:

- `app` startup waits for Postgres readiness and runs migrations automatically.
- The configured Postgres image (`pgvector/pgvector:pg16`) includes pgvector, and the initial migration creates the `vector` extension.

## Top-Level Architecture

Single entry point with intent-based routing to 4 parallel paths. Each path handles a specific user intent (URL extraction, text extraction, recipe search, or meal planning) and converges to an appropriate final state.

### Unified Graph

```mermaid
graph TD
    user["User input / request"] --> router{Intent Router<br/>node}

    router -- "URL" --> pathA[Path A: URL Extraction]
    router -- "Recipe text" --> pathB[Path B: Text Extraction]
    router -- "Search recipe" --> pathC[Path C: Recipe Search]
    router -- "Meal plan" --> pathD[Path D: Meal Planning]

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

    router -- URL --> fetchUrl
    router -- "Text recipe" --> extractRecipe2

    %% Path C — Recipe search
    subgraph "C. Recipe Search"
        searchDb["search_db_with_vectors<br/>(pgvector cosine distance)"] --> returnRecipe["return_recipe<br/>(formatted response)"]
    end

    router -- "Search request" --> searchDb

    %% Path D — Meal plan generation
    subgraph "D. Meal Planning"
        evalPlan["evaluate_plan<br/>→ check if user provided<br/>ingredients, restrictions,<br/>preferences"]
        evalPlan -- insufficient --> askUser["ask_user<br/>→ request missing info"]
        askUser --> evalPlan
        evalPlan -- sufficient --> genMealPlan["generate_meal_plan<br/>(LLM)"]
        genMealPlan --> generateEmbeddings3["generate_embeddings"]
        generateEmbeddings3 --> generateLabels3["generate_labels"]
        generateLabels3 --> persistD["persist to db<br/>(database_upsert)"]
    end

    router -- "Meal plan request" --> evalPlan

    %% Final nodes
    persistA --> persistEnd1["✅ Persisted to DB"]
    persistB --> persistEnd2["✅ Persisted to DB"]
    returnRecipe --> searchEnd["✅ Recipe returned<br/>to user"]
    persistD --> mealEnd["✅ Meal plan persisted"]

    classDef startNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef routerNode fill:#fff9c4,stroke:#f9a825,stroke-width:2px;
    classDef toolNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef llmNode fill:#fce4ec,stroke:#ad1457,stroke-width:1px,stroke-dasharray:5 5;
    classDef decision fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef finalNode fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px;

    class user startNode;
    class router decision;
    class pathA,pathB,pathC,pathD llmNode;
    class fetchUrl,parseHtml,persistA,persistB,persistD,searchDb,returnRecipe,askUser toolNode;
    class extractRecipe2,llmNode finalNode;
    class persistEnd1,persistEnd2,searchEnd,mealEnd finalNode;
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

## Path D — Meal Planning Detail (with iterative loop)

**Trigger:** User requests a weekly meal plan.
**Flow:** Evaluate user info → ask if insufficient → generate plan → embeddings & labels → persist.

```mermaid
graph TD
    router{Intent: Meal<br/>plan?} -- yes --> evalInfo["evaluate_user_info"]
    router -- no --> other["→ other path"]

    evalInfo -- "missing info" --> askUser["ask_user<br/>(ingredients,<br/>restrictions,<br/>preferences)"]
    askUser --> userReply["user provides<br/>additional info"]
    userReply --> evalInfo

    evalInfo -- "sufficient info" --> genPlan["generate_meal_plan<br/>(LLM + database)<br/>→ 7-day plan"]

    genPlan --> mealEmbeddings["generate_embeddings"]
    mealEmbeddings --> mealLabels["generate_labels<br/>(meal_type, day,<br/>tags)"]
    mealLabels --> persistMeal["persist to db<br/>(database_upsert)"]

    persistMeal --> planSaved["✅ Meal plan persisted"]

    classDef routerNode fill:#fff9c4,stroke:#f9a825,stroke-width:2px;
    classDef toolNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef llmNode fill:#fce4ec,stroke:#ad1457,stroke-width:1px,stroke-dasharray:5 5;
    classDef decision fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef finalNode fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px;

    class router,evalInfo decision;
    class genPlan,llmNodeB llmNode;
    class askUser,searchDb toolNode;
    class planSaved finalNode;
```

---

## Summary Table (Updated)

| Path | Trigger | Node Sequence | End State |
|------|---------|---------------|-----------|
| **A** — URL Extraction | User submits a URL | `fetch_url` → `parse_html` → `extract_recipe` → `generate_embeddings` → `generate_labels` → `persist to db` | Persisted to DB |
| **B** — Text Extraction | User pastes recipe text | `extract_recipe` → `generate_embeddings` → `generate_labels` → `persist to db` | Persisted to DB |
| **C** — Recipe Search | User searches for a recipe | `parse_query` → `search_db_with_vectors` → `format_response` / `empty response` | Returned to user |
| **D** — Meal Planning | User requests a meal plan | `evaluate_user_info` → _(loop: ask if insufficient)_ → `generate_meal_plan` → `generate_embeddings` → `generate_labels` → `persist to db` | Persisted to DB |
