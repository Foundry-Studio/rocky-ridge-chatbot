FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

WORKDIR /app
COPY pyproject.toml ./
COPY README.md ./
RUN pip install -e . --no-cache-dir

COPY . .

RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -fsSL http://localhost:${PORT:-8000}/healthz || exit 1

# Hard single-worker: Chainlit Socket.IO + JSONL append both assume
# one process. DO NOT add --workers.
ENTRYPOINT ["scripts/entrypoint.sh"]
