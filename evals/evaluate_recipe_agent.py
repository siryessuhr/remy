"""LangSmith evaluation script for the Remy recipe agent.

This script uploads a local dataset and evaluates the agent against a small set of
high-signal checks: intent routing, recipe fidelity, and search relevance.

The script is designed to work with the actual Remy app flow defined in
src/remy/agents/recipe_agent.py and the dataset JSON in this directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client, aevaluate
from loguru import logger as log

from remy.agents.recipe_agent import RecipeAgent

load_dotenv()

DATASET_PATH = Path(__file__).with_name("recipe_agent_dataset_15.json")
DATASET_NAME = "remy-recipe-agent-eval"


def _example_to_dict(example: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert LangSmith Example objects or raw dicts to input/output dicts."""
    if isinstance(example, dict):
        inputs = example.get("inputs", {})
        outputs = example.get("outputs", {})
    else:
        inputs = getattr(example, "inputs", None) or {}
        outputs = getattr(example, "outputs", None) or {}

    if hasattr(inputs, "dict"):
        inputs = inputs.dict()
    if hasattr(outputs, "dict"):
        outputs = outputs.dict()

    if not isinstance(inputs, dict):
        inputs = {}
    if not isinstance(outputs, dict):
        outputs = {}

    return inputs, outputs


def _normalize_output(payload: Any) -> dict[str, Any]:
    """Normalize LangSmith target output into a dict for scoring.

    Args:
        payload: Raw output returned by the agent.

    Returns:
        A dict-friendly representation of the agent output.
    """
    if isinstance(payload, dict):
        if "output" in payload and isinstance(payload["output"], dict):
            return payload["output"]
        return payload
    if isinstance(payload, str):
        return {"message": payload}
    return {"value": payload}


async def run_recipe_agent_on_example(example: Any) -> dict[str, Any]:
    """Invoke the actual Remy agent for a single evaluation example.

    Args:
        example: A LangSmith example object or a raw dict containing ``inputs`` and ``outputs``.

    Returns:
        The agent's raw output payload.
    """
    inputs, _ = _example_to_dict(example)
    user_request = str(inputs.get("user_request", "")).strip()

    if not user_request:
        return {"user_intent": "search_recipe_in_db", "search_response": {"message": "No input provided."}}

    agent = RecipeAgent.from_env()

    if agent.session_factory is None:
        return {
            "user_intent": "search_recipe_in_db",
            "search_response": {"message": "No DB session factory configured."},
        }

    async with agent.session_factory() as session:  # type: ignore[attr-defined]
        result = await agent.invoke(user_request, session=session)

    if isinstance(result, dict):
        return _normalize_output(result)

    return {"value": str(result)}


def _expected_intent(example: Any) -> str | None:
    """Fetch the expected intent from an example."""
    _, outputs = _example_to_dict(example)
    return outputs.get("user_intent")


def _actual_intent(run: Any) -> str | None:
    """Extract the agent intent from a run output."""
    output = _normalize_output(getattr(run, "outputs", None) or getattr(run, "output", None) or {})
    return output.get("user_intent")


def metric_intent_accuracy(run: Any, example: dict[str, Any]) -> dict[str, Any]:
    """Score whether the agent picked the correct route.

    This is one of the most important demo metrics because the graph behavior is
    defined by the route chosen in the initial intent step.
    """
    expected = _expected_intent(example)
    actual = _actual_intent(run)
    score = 1.0 if expected == actual else 0.0
    return {
        "key": "intent_accuracy",
        "score": score,
        "comment": f"expected={expected}, actual={actual}",
    }


def metric_recipe_fidelity(run: Any, example: Any) -> dict[str, Any]:
    """Score recipe extraction quality when a recipe is expected.

    This uses a simple structured check: if the example expects a recipe, the model
    should output one with the key fields populated. If the example does not expect
    a recipe, the output should be empty for the recipe fields.
    """
    _, outputs = _example_to_dict(example)
    expected_recipe = outputs.get("processed_recipe")
    output = _normalize_output(getattr(run, "outputs", None) or getattr(run, "output", None) or {})
    actual_recipe = output.get("processed_recipe")

    if expected_recipe is None:
        score = 1.0 if actual_recipe in (None, {}) else 0.0
        return {
            "key": "recipe_fidelity",
            "score": score,
            "comment": "No recipe expected for this example.",
        }

    if not isinstance(actual_recipe, dict):
        return {
            "key": "recipe_fidelity",
            "score": 0.0,
            "comment": "Recipe output missing or malformed.",
        }

    required_keys = {"title", "ingredients", "instructions"}
    present = required_keys.issubset(actual_recipe.keys())
    title_matches = str(actual_recipe.get("title", "")).strip() == str(expected_recipe.get("title", "")).strip()
    score = 1.0 if present and title_matches else 0.0
    return {
        "key": "recipe_fidelity",
        "score": score,
        "comment": f"title_match={title_matches}, keys_present={present}",
    }


def metric_search_relevance(run: Any, example: Any) -> dict[str, Any]:
    """Score retrieval relevance against expected search results.

    This checks whether the selected top titles overlap with the expected results.
    """
    _, outputs = _example_to_dict(example)
    expected_results = outputs.get("search_results", [])
    output = _normalize_output(getattr(run, "outputs", None) or getattr(run, "output", None) or {})
    actual_results = output.get("search_results", [])

    if not expected_results:
        score = 1.0 if not actual_results else 0.0
        return {
            "key": "search_relevance",
            "score": score,
            "comment": "No retrieval expected for this example.",
        }

    expected_titles = {str(item.get("title", "")).lower().strip() for item in expected_results if item.get("title")}
    actual_titles = {str(item.get("title", "")).lower().strip() for item in actual_results if item.get("title")}
    if not expected_titles:
        return {
            "key": "search_relevance",
            "score": 1.0 if not actual_titles else 0.0,
            "comment": "No expected titles available.",
        }

    overlap = len(expected_titles & actual_titles)
    score = overlap / max(len(expected_titles), 1)
    return {
        "key": "search_relevance",
        "score": score,
        "comment": f"overlap={overlap}, expected_titles={sorted(expected_titles)}",
    }


def load_local_examples(path: Path) -> list[dict[str, Any]]:
    """Load the local JSON dataset.

    Args:
        path: Path to the dataset JSON file.

    Returns:
        A list of examples suitable for LangSmith.
    """
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if not isinstance(data, list):
        message = "Dataset JSON must be a list of examples."
        raise TypeError(message)

    return data


def upload_dataset_to_langsmith(dataset_name: str, examples: list[dict[str, Any]]) -> Any:
    """Upload the local examples to LangSmith as a dataset.

    Args:
        dataset_name: Name to assign in LangSmith.
        examples: The local examples to upload.

    Returns:
        The created LangSmith dataset object.
    """
    client = Client()
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Remy recipe agent evaluation dataset covering extraction, search, and meal-plan routing.",
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": example["inputs"],
                "outputs": example["outputs"],
            }
            for example in examples
        ],
    )
    log.info("Uploaded {} examples to LangSmith dataset '{}'", len(examples), dataset_name)
    return dataset


async def evaluate_target(example: Any) -> dict[str, Any]:
    """Target function for LangSmith evaluation.

    This must be async so the coroutine is awaited by the LangSmith runner.
    """
    return await run_recipe_agent_on_example(example)


async def run_langsmith_evaluation(dataset_name: str) -> Any:
    """Run the LangSmith evaluation against the uploaded dataset.

    Args:
        dataset_name: Dataset name previously created in LangSmith.

    Returns:
        The LangSmith evaluation result.
    """
    return await aevaluate(
        evaluate_target,
        data=dataset_name,
        evaluators=[
            metric_intent_accuracy,
            metric_recipe_fidelity,
            metric_search_relevance,
        ],
        max_concurrency=2,
    )


async def main() -> None:
    """Upload the local dataset and run LangSmith evaluation."""
    examples = load_local_examples(DATASET_PATH)
    upload_dataset_to_langsmith(DATASET_NAME, examples)
    await run_langsmith_evaluation(DATASET_NAME)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
