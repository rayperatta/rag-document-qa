# RAG Document Q&A API

A production-ready Retrieval-Augmented Generation system for querying PDF documents using natural language.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## What it does

1. Upload PDF documents via web UI or API
2. Documents are chunked and embedded using sentence-transformers (local, free)
3. Embeddings stored in ChromaDB (local vector database)
4. Ask questions in natural language → retrieves relevant chunks → LLM generates answer with source citations

## Stack

- **FastAPI** — REST API + web UI
- **LangChain** — RAG orchestration, document chunking
- **ChromaDB** — vector database (local, persistent)
- **sentence-transformers** — local embeddings (no API key needed)
- **OpenRouter / OpenAI** — LLM for answer generation (configurable)

## Quick start

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure LLM (optional — works with local embeddings alone)
cp .env.example .env
# Edit .env with your OpenRouter API key

# 3. Run
uvicorn main:app --reload --port 8000

# 4. Open http://localhost:8000
```

## Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload PDF document(s) |
| `POST` | `/api/ask` | Ask a question (returns answer + sources) |
| `GET` | `/api/documents` | List uploaded documents |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `GET` | `/api/health` | Health check |

## Architecture

```
[PDF Upload] → [PyPDF Loader] → [Text Splitter] → [Embeddings]
                                                        ↓
                                                 [ChromaDB]
                                                        ↓
[User Question] → [Embeddings] → [Vector Search] → [Top-K Chunks]
                                                        ↓
                                             [LLM Prompt + Context]
                                                        ↓
                                                  [Answer + Sources]
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Lint
ruff check .
```

## Environment variables

See `.env.example` for all configurable options.

## Author

Ray Peratta — [GitHub](https://github.com/gozuray) — [LinkedIn](https://www.linkedin.com/in/ray-peratta)
