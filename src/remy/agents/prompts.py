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
