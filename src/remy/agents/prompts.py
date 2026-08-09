"""Various prompts used by Remy agent steps."""

RECIPE_EXTRACTION_PROMPT = """
Extract a recipe from the following text input. Respond with a valid JSON object only.
The JSON object should have the following structure:
```json
{{
    "url": "string",
    "title": "string",
    "ingredients": "string",
    "instructions": "string",
    "labels": "string"
}}
```
Text input:
{text}
"""

USER_INTENT_PROMPT = """
You are a recipe extraction agent. You will be given a user request and you need to determine the user's intent.

The possible intents are:
1. extract_recipe_from_url: The user wants to extract a recipe from a given URL.
2. extract_recipe_from_text: The user wants to extract a recipe from a given text input.
3. search_recipe_in_db: The user wants to search for a recipe in the database
4. generate_meal_plan: The user wants to generate a meal plan based on their dietary preferences and restrictions.

Return a JSON object with the following structure:
{{
    "user_intent": "extract_recipe_from_url",
    "url": "string"
}}

If there is no URL provided, the "url" field should be an empty string.

The user request is: "{user_request}".
"""
