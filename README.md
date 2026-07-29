# RAG Document Q&A API

A production-ready Retrieval-Augmented Generation system for querying PDF documents using natural language.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## What it does

1. Upload PDF documents via web UI or API
2. Documents are chunked and embedded using sentence-transformers (local, free)
3. Embeddings stored in ChromaDB (local vector database)
4. Ask questions in natural language → **hybrid retrieval** (BM25 + vector + cross-encoder reranking) → LLM generates answer with source citations
5. Every query is **traced with Langfuse** (retrieval spans, generation spans, token usage, latency)
6. Pipeline quality is **measured with RAGAS** (faithfulness, context precision/recall, answer relevancy)

## Stack

- **FastAPI** — REST API + web UI
- **LangChain** — RAG orchestration, document chunking
- **ChromaDB** — vector database (local, persistent)
- **sentence-transformers** — local embeddings + cross-encoder reranker (no API key needed)
- **rank-bm25** — keyword search for hybrid retrieval
- **OpenRouter / OpenAI** — LLM for answer generation (configurable)
- **Langfuse** — observability and tracing (optional, self-hosted)
- **RAGAS** — evaluation metrics (dev dependency)

## Key features

### Hybrid search + reranking
Pure vector search misses exact terms (proper nouns, error codes, numbers). The hybrid retriever combines:
- **BM25** keyword search (in-memory, synced with ChromaDB)
- **Vector search** (ChromaDB cosine similarity)
- **Reciprocal Rank Fusion (RRF)** to merge rankings without comparable score scales
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for final precision

Toggle with `HYBRID_SEARCH=true` and `RERANKER_ENABLED=true`.

### Observability (Langfuse)
Every `/api/ask` call produces a trace with:
- A **retrieval span** (query, chunks, scores, latency, mode)
- A **generation span** (model, prompt, answer, token usage, latency)

No-op when Langfuse keys are not set — the app runs identically without an observability backend.

### Evaluation (RAGAS)
Answer the interview question *"how do you know your RAG works well?"* with numbers:

```bash
pip install -r requirements-dev.txt
# Upload documents, then:
python -m eval.run_eval --dataset eval/golden_dataset.json --k 5
```

Metrics: **faithfulness** (anti-hallucination), **answer relevancy**, **context precision**, **context recall**.

The golden dataset (`eval/golden_dataset.json`) is versioned JSON — run before/after any retrieval change and compare.

### CI/CD
GitHub Actions runs `ruff check` + `pytest` on every push and PR (`.github/workflows/ci.yml`).

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
| `GET` | `/api/health` | Health check (includes hybrid + tracing status) |

## Architecture

```
[PDF Upload] → [PyPDF Loader] → [Text Splitter] → [Embeddings]
                                                        ↓
                                                 [ChromaDB]
                                                        ↓
[User Question] → [BM25 Search] ──┐                    │
              → [Vector Search] ─┤                    │
                                   ↓                    │
                          [Reciprocal Rank Fusion] ←────┘
                                   ↓
                         [Cross-Encoder Reranker]
                                   ↓
                          [Top-K Chunks]
                                   ↓
                          [LLM Prompt + Context]
                                   ↓
                          [Answer + Sources]
                                   ↓
                        [Langfuse Trace]
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Lint
ruff check .

# Evaluate pipeline quality (requires OPENROUTER_API_KEY + uploaded docs)
python -m eval.run_eval
```

## Environment variables

See `.env.example` for all configurable options.

## Author

Ray Peratta — [GitHub](https://github.com/gozuray) — [LinkedIn](https://www.linkedin.com/in/prince-peratta)
