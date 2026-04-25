FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

WORKDIR /app
# pyproject.toml uses package-dir = src, so setuptools requires src/ at install
# time. Copy everything first, then install — layer-caching optimization isn't
# available for editable installs over a src/ layout.
COPY . .
RUN pip install --no-cache-dir .

RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -fsSL http://localhost:${PORT:-8000}/healthz || exit 1

# Hard single-worker: Chainlit Socket.IO + JSONL append both assume
# one process. DO NOT add --workers.
ENTRYPOINT ["scripts/entrypoint.sh"]
