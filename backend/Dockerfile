FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
RUN mkdir -p src/rag_persona && touch src/rag_persona/__init__.py && pip install .
COPY src ./src
RUN pip install . --no-deps

CMD uvicorn rag_persona.main:app --host 0.0.0.0 --port ${PORT:-8000}


