# Rocky Ridge Chatbot

Grounded RAG web chatbot for Rocky Ridge Land Management's knowledge base. First-deployment target of the Venture Knowledge Chatbot (VKC) — a sibling Railway service to [Foundry-Agent-System](https://github.com/Foundry-Studio/Foundry-Agent-System) that answers questions strictly from a tenant's ingested knowledge library with inline source citations.

**Status:** v0 demo. Built for the Rocky Ridge client demo to Patience Knight (Alabama A&M).

## Quickstart (local dev)

1. **Python 3.11.** Confirm `python --version`.
2. Copy `.env.example` → `.env`. Fill in `FOUNDRY_INTERNAL_TOKEN` — value from the Foundry-Agent-System Railway `CHATBOT_INTERNAL_TOKEN` env var.
3. Install:
   ```bash
   pip install -e ".[dev]"
   ```
4. Smoke test against live Foundry (must be VPN-accessible or prod URL):
   ```bash
   python scripts/smoke.py "what is canebrake restoration"
   ```
5. Serve UI:
   ```bash
   chainlit run src/chatbot/app.py
   ```
   Open http://localhost:8000.

## Testing

```bash
pytest                                       # full unit + mocked integration
pytest tests/test_foundry_client.py -v       # one module verbose
```

## Architecture (brief)

```
browser ─(Socket.IO)─▶ Chainlit (this repo)
                          │
                          ├─ POST /api/v1/knowledge/search
                          └─ POST /api/v1/roster/v1/chat/completions (SSE)
                              ▼
                        Foundry-Agent-System (sibling Railway service)
                              │
                              ├─ Postgres (BM25)
                              ├─ Pinecone (vector)
                              └─ Anthropic Sonnet 4.5 via LLM Roster
```

- **LLM:** `anthropic/claude-sonnet-4-5` for every call (answer + follow-up reformulation).
- **History:** last 6 turns, retrieved chunks stripped from history.
- **Refusal:** RRF score normalized to 0–1; refuses if no chunk scores ≥ `CHATBOT_REFUSAL_THRESHOLD` (default 0.3).
- **Citations:** inline `[ref:<id>]` markers resolve to Chainlit side-panel elements. Unmatched markers are stripped (G6 integrity).
- **Logs:** JSONL on Railway volume, 100 MB rotation, 30-day retention.
- **Cost control:** 5 req/min/IP + `$50/day` circuit breaker.

## Governing documents

- **Plan:** [Foundry-Agent-System/WORKBENCH/tim/projects/agents/venture-knowledge-chatbot/plan-draft/IMPLEMENTATION-PLAN.md](https://github.com/Foundry-Studio/Foundry-Agent-System/blob/master/WORKBENCH/tim/projects/agents/venture-knowledge-chatbot/plan-draft/IMPLEMENTATION-PLAN.md)
- **PM project:** VKC (`0cc745da-4ccb-4f9b-a9ca-1fa44224baa5`)
- **D-numbers locked:** D-136 (dedicated token), D-137 (actor-tenant authz), D-138 (VKC risk acceptance)
- **Parent system context:** [Foundry-Agent-System/CLAUDE.md](https://github.com/Foundry-Studio/Foundry-Agent-System/blob/master/CLAUDE.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
