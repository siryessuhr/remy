#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

wait_for_database() {
	echo "Waiting for database to become available..."

	attempts=0
	max_attempts=30

	while [ "$attempts" -lt "$max_attempts" ]; do
		if uv run python - <<'PY'
import os
import sys

import psycopg2
from sqlalchemy.engine import make_url


def normalize_db_url(raw_url: str) -> str:
		url = make_url(raw_url)
		if "+" in url.drivername:
				base_driver = url.drivername.split("+", 1)[0]
				url = url.set(drivername=base_driver)
		return url.render_as_string(hide_password=False)


database_url = os.getenv("DATABASE_URL")
if not database_url:
		sys.exit(1)

try:
		conn = psycopg2.connect(normalize_db_url(database_url))
		conn.close()
except Exception:
		sys.exit(1)
PY
		then
			echo "Database is available."
			return 0
		fi

		attempts=$((attempts + 1))
		echo "Database not ready yet (attempt $attempts/$max_attempts)."
		sleep 2
	done

	echo "Database did not become available in time."
	return 1
}

if [ -z "$DATABASE_URL" ]; then
	echo "DATABASE_URL is not set."
	exit 1
fi

wait_for_database

echo "Running database migrations..."
uv run python -m remy migrate upgrade head


PORT=${PORT:-8000}
echo "Starting uvicorn on port $PORT..."
exec uv run uvicorn remy.app:app --port "$PORT" --host 0.0.0.0
