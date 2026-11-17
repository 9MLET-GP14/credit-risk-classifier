# Imagem base com Python 3.12 e UV pré-instalado
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=0 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "scripts/run_pipeline.py", "--scenario", "drift"]

