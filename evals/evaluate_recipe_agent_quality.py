"""LangSmith quality evaluation for Remy recipe-agent outputs.

This script adds a second evaluator focused on quality dimensions that are less
structural and more product-facing: label quality and hallucination risk. It is
intended to complement the first evaluator in evaluate_recipe_agent.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client, aevaluate
from loguru import logger as log

load_dotenv()

DATASET_PATH = Path(__file__).with_name("recipe_agent_dataset_15.json")
DATASET_NAME = "remy-recipe-agent-quality"


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
    """Normalize a raw run output into a dict."""
    if isinstance(payload, dict):
        if "output" in payload and isinstance(payload["output"], dict):
            return payload["output"]
        return payload
    if isinstance(payload, str):
        return {"message": payload}
    return {"value": str(payload)}


def _tokenize(value: str) -> set[str]:
    """Tokenize a free-form string into lowercase word fragments."""
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token}


def _expected_labels(example: Any) -> set[str]:
    """Return the expected labels for an example, if provided."""
    _, outputs = _example_to_dict(example)
    labels = outputs.get("labels", [])
    return {str(label).lower().strip() for label in labels if str(label).strip()}


def _actual_labels(run: Any) -> set[str]:
    """Return labels present in the run output."""
    output = _normalize_output(getattr(run, "outputs", None) or getattr(run, "output", None) or {})
    labels = output.get("labels", [])
    if isinstance(labels, str):
        labels = [labels]
    return {str(label).lower().strip() for label in labels if str(label).strip()}


def metric_label_quality(run: Any, example: Any) -> dict[str, Any]:
    """Measure whether generated labels match the expected recipe labels."""
    expected = _expected_labels(example)
    actual = _actual_labels(run)

    if not expected:
        score = 1.0 if not actual else 0.0
        return {
            "key": "label_quality",
            "score": score,
            "comment": "No labels expected for this example.",
        }

    intersection = len(expected & actual)
    score = intersection / max(len(expected), 1)
    return {
        "key": "label_quality",
        "score": score,
        "comment": f"expected={sorted(expected)}, actual={sorted(actual)}",
    }


def metric_hallucination_risk(run: Any, example: Any) -> dict[str, Any]:
    """Measure the risk of invented recipe details.

    This uses a conservative lexical consistency check: when a recipe is expected,
    the actual recipe must share a meaningful amount of ingredient vocabulary with the
    expected recipe. This helps catch hallucinated or unrelated recipes.
    """
    _, outputs = _example_to_dict(example)
    expected_recipe = outputs.get("processed_recipe")
    output = _normalize_output(getattr(run, "outputs", None) or getattr(run, "output", None) or {})
    actual_recipe = output.get("processed_recipe")

    if expected_recipe is None:
        score = 1.0 if actual_recipe in (None, {}) else 0.0
        return {
            "key": "hallucination_risk",
            "score": score,
            "comment": "No recipe expected for the example.",
        }

    if not isinstance(actual_recipe, dict):
        return {
            "key": "hallucination_risk",
            "score": 0.0,
            "comment": "Recipe output missing or malformed.",
        }

    expected_ingredients = _tokenize(str(expected_recipe.get("ingredients", "")))
    actual_ingredients = _tokenize(str(actual_recipe.get("ingredients", "")))

    if not expected_ingredients:
        return {
            "key": "hallucination_risk",
            "score": 1.0,
            "comment": "No ingredient set provided for expected recipe.",
        }

    if not actual_ingredients:
        return {
            "key": "hallucination_risk",
            "score": 0.0,
            "comment": "Actual recipe ingredients missing.",
        }

    overlap = len(expected_ingredients & actual_ingredients)
    score = overlap / max(len(expected_ingredients), 1)
    return {
        "key": "hallucination_risk",
        "score": score,
        "comment": f"ingredient_overlap={overlap}, expected_ingredients={sorted(expected_ingredients)}",
    }


def load_local_examples(path: Path) -> list[dict[str, Any]]:
    """Load the local dataset JSON."""
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if not isinstance(data, list):
        message = "Dataset JSON must be a list of example objects."
        raise TypeError(message)

    return data


def upload_dataset_to_langsmith(dataset_name: str, examples: list[dict[str, Any]]) -> Any:
    """Upload examples to LangSmith as a dataset."""
    client = Client()
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Remy recipe agent quality evals focused on label quality and hallucination risk.",
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


async def quality_target(example: Any) -> dict[str, Any]:
    """Target function for LangSmith quality evaluation."""
    _, outputs = _example_to_dict(example)
    return {
        "user_intent": outputs.get("user_intent"),
        "labels": outputs.get("labels", []),
        "processed_recipe": outputs.get("processed_recipe"),
    }


async def run_quality_evaluation(dataset_name: str) -> Any:
    """Run the quality evaluation suite in LangSmith."""
    return await aevaluate(
        quality_target,
        data=dataset_name,
        evaluators=[
            metric_label_quality,
            metric_hallucination_risk,
        ],
        max_concurrency=2,
    )


async def main() -> None:
    """Upload the dataset and run the quality-focused evaluator."""
    examples = load_local_examples(DATASET_PATH)
    upload_dataset_to_langsmith(DATASET_NAME, examples)
    await run_quality_evaluation(DATASET_NAME)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
