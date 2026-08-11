#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Running database migrations..."
uv run --no-dev python -m remy migrate upgrade head

PORT=${PORT:-8000}
echo "Starting uvicorn on port $PORT..."
exec uv run --no-dev uvicorn remy.app:app --port "$PORT" --host 0.0.0.0
