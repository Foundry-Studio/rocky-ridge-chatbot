#!/bin/bash
set -euo pipefail

# Render tenant-branded chainlit config + welcome at container start
# (Railway env vars are runtime — templates read them here).
python scripts/render_config.py

exec chainlit run src/chatbot/app.py --host 0.0.0.0 --port "${PORT:-8000}" --headless -w
