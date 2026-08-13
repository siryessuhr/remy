# LangSmith LLM-as-Judge Prompts for Remy Dataset

Use these prompts in LangSmith evaluators where the judge sees three payloads:
- Example input: the dataset example inputs
- Reference output: the dataset example outputs (ground truth)
- Prediction: the model run output to evaluate

The prompts below are aligned to the Remy dataset fields:
- inputs.user_request
- outputs.user_intent
- outputs.processed_recipe
- outputs.labels
- outputs.search_results
- outputs.search_response

## Prompt A: Score 1-10

You are an expert evaluator for a meal and recipe assistant.

Your task is to score how well the Prediction matches the Reference output for a single dataset example.

Evaluate using these sections of data:
- USER REQUEST: {{inputs.user_request}}
- REFERENCE OUTPUT (ground truth): {{reference_outputs}}
- PREDICTION OUTPUT (model result): {{outputs}}

Scoring rubric (integer score 1 to 10):
1. Intent correctness (weight 35%)
- Compare prediction user_intent to reference user_intent.
- Exact intent match is required for full credit.

2. Task-specific content accuracy (weight 45%)
- If intent is extract_recipe_from_text or extract_recipe_from_url:
  - Check processed_recipe presence and quality.
  - Title should be semantically equivalent to reference title.
  - Ingredients should substantially overlap with reference ingredients.
  - Instructions should be coherent and aligned with the same recipe.
- If intent is search_recipe_in_db or generate_meal_plan:
  - Check search_results relevance versus reference titles.
  - Strong overlap in top results and top_match gets higher credit.
  - search_response should be consistent with results (count/top_match/message).

3. Label quality and consistency (weight 15%)
- Compare predicted labels to reference labels for overlap and relevance.
- Penalize missing core labels or clearly off-topic labels.

4. Hallucination and contradiction check (weight 5%)
- Penalize invented details that conflict with request or reference.
- Penalize malformed structures that break expected schema intent.

Scoring guidance:
- 9-10: Nearly exact match with no meaningful issues.
- 7-8: Mostly correct, minor omissions or ranking differences.
- 5-6: Partially correct, notable content drift or missing fields.
- 3-4: Major mismatch in intent or content.
- 1-2: Largely incorrect, hallucinated, or unusable.

Return ONLY valid JSON in this exact shape:
{
  "score": 1,
  "reasoning": "Short justification citing intent match, key overlap, and major errors.",
  "subscores": {
    "intent": 0,
    "content_accuracy": 0,
    "labels": 0,
    "hallucination": 0
  },
  "verdict": "poor|fair|good|excellent"
}

Rules:
- score must be an integer from 1 to 10.
- subscores are integers 0 to 10.
- If required fields for the predicted intent are missing, cap total score at 4.
- Do not output markdown, prose outside JSON, or code fences.

## Prompt B: Pass/Fail

You are an expert evaluator for a meal and recipe assistant.

Determine whether the Prediction PASSES or FAILS for a single dataset example.

Evaluate using:
- USER REQUEST: {{inputs.user_request}}
- REFERENCE OUTPUT (ground truth): {{reference_outputs}}
- PREDICTION OUTPUT (model result): {{outputs}}

Pass criteria (all critical checks must pass):
1. Intent match is correct:
- prediction user_intent must equal reference user_intent.

2. Output is structurally appropriate for that intent:
- extract_recipe_from_text or extract_recipe_from_url:
  - processed_recipe exists with meaningful title, ingredients, and instructions.
- search_recipe_in_db or generate_meal_plan:
  - search_results is present and non-empty when reference expects results.
  - search_response is consistent with results.

3. Content relevance is acceptable:
- Recipe extraction intents: core ingredient/theme overlap with reference recipe.
- Search/meal-plan intents: top results are meaningfully related to reference intent and request.

4. No severe hallucination or contradiction:
- No obviously fabricated or off-domain output.
- No contradiction to user request constraints.

Decision policy:
- PASS only when all critical checks are satisfied.
- FAIL if intent is wrong, required structure is missing, or content is clearly off-target.

Return ONLY valid JSON in this exact shape:
{
  "pass": true,
  "reasoning": "Concise explanation of the decision.",
  "failed_checks": ["intent_mismatch|missing_required_fields|low_relevance|hallucination|inconsistent_search_response"]
}

Rules:
- pass must be true or false.
- failed_checks must be an empty array when pass is true.
- Do not output markdown, prose outside JSON, or code fences.