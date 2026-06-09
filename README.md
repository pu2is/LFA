# LFA — Local File Assistant

A fully offline document management tool that uses a local LLM to help users organize and search PDF/Word files stored on their machine.

Users point the app at local folders, the LLM suggests labels for each document, and users can later search their files using natural language.

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL + pgvector, Redis + RQ, LangChain, Ollama (TinyLLaMA)
- **Infrastructure**: Docker Compose (PostgreSQL, Redis, Ollama)
- **Frontend**: React + Tailwind CSS _(planned)_
