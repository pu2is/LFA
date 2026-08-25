# LFA — Local File Assistant

A fully offline document management tool that uses a local LLM to help users organize and search PDF/Word files stored on their machine.

Users point the app at local folders, the LLM suggests labels for each document, and users can later search their files using natural language.

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL + pgvector, Redis + RQ, LangChain, Ollama
- **Infrastructure**: Docker Compose (PostgreSQL, Redis, Ollama)
- **Frontend**: React + Tailwind CSS _(planned)_

## Getting Started

### Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.12 (see `backend/.python-version`)
- Docker (runs PostgreSQL, Redis, Ollama)

### Setup

1. Start the infrastructure services:
   ```
   docker compose up -d
   ```
   This also pulls the `qwen2.5:3b` and `bge-m3` Ollama models automatically.
2. Create `backend/.env`:
   ```env
   DATABASE_URL=postgresql+psycopg://lfa:lfa_password@localhost:5432/lfa
   REDIS_URL=redis://localhost:6379/0
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=qwen2.5:3b
   OLLAMA_EMBED_MODEL=bge-m3
   ```
3. Start the FastAPI backend (with hot reload):
   ```
   cd backend && uv run fastapi dev app/main.py
   ```
4. Start the RQ worker in a separate terminal (on Windows, use `SimpleWorker` — see [docs/99_dev-setup.md](docs/99_dev-setup.md)):
   ```
   uv run rq worker --worker-class rq.SimpleWorker --url redis://localhost:6379/0 label ingest scan embed
   ```
5. Open `http://localhost:8000/docs` for the auto-generated Swagger UI.

For Windows-specific worker notes and the optional LibreOffice setup (legacy `.doc` support), see [docs/99_dev-setup.md](docs/99_dev-setup.md).
