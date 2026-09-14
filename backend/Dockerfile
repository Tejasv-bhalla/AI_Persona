FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies in their own layer so source edits do not invalidate the
# (slow) dependency install. The stub package satisfies the build backend.
COPY pyproject.toml ./
RUN mkdir -p src/rag_persona \
    && touch src/rag_persona/__init__.py \
    && pip install --no-cache-dir .

COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Drop root before running the service.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn rag_persona.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
