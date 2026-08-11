---
applyTo: '**/*.py'
---
You always include type hints for function parameters and return types.
You always write Python code using Python 3.13, including using the latest syntax features.
You follow best practices and write clean, maintainable, extensible code.
Clean, maintainable, extensible code includes small testable functions, descriptive variable names, and clear logic.
Cognitive complexity should be kept very low.
Docstrings should be used for all functions to describe their purpose, using Google docstring format.
Add comments to help explain complex or complicated logic, but do not add comments for simple or obvious code.
Never add comments to the code explaining what just changed.
Do not use the keywords `global` or `nonlocal`.
Never use print statements and always use logging by importing the loguru module. `from loguru import logger as log` and used as `log.info("Your log message")`.
You always follow SOLID principles.
Ensure code is DRY. Common code should be in separate functions or classes.
Do not try to fix formatting isues in the code, you can use `uv run ruff format`, `uv run ruff check --fix` or the shortcut `poe fmt`.
Do not try and do everything all at once. Walk the user through the process step by step.
Ask the user clarifying questions if needed rather than making assumptions.
Never keep "original" files or versioned files in PRs (e.g., file_original.py, file_v1.py, file_backup.py, etc)
