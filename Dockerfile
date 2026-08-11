FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS build

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV DEBIAN_FRONTEND=noninteractive

RUN useradd --create-home --uid 1001 --shell /bin/bash appuser
USER appuser

WORKDIR /home/appuser/app

ENV PATH="$PATH:/home/appuser/app/.venv/bin"

RUN --mount=type=cache,target=/home/appuser/app/.cache/uv,uid=1001 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

FROM build AS service

COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./
COPY src/ ./src/
COPY scripts/start_server.sh ./scripts/

RUN --mount=type=cache,target=/home/appuser/app/.cache/uv,uid=1001 \
    uv sync --frozen --no-dev

EXPOSE 8000

CMD ["/home/appuser/app/scripts/start_server.sh"]
