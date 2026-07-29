"""LLM answer generation module — OpenRouter API integration."""
import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer the user's question based ONLY on the provided context chunks. If the answer is not in the context, say "I couldn't find this information in the uploaded documents." Always cite which chunk(s) you used."""

USER_TEMPLATE = """Context chunks from uploaded documents:

{context}

---

Question: {question}

Answer based only on the above context. Be concise and cite sources."""


class LLMGenerator:
    """Generate answers using an OpenRouter-compatible LLM API."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct:free"):
        """Initialize the generator.

        Args:
            api_key: OpenRouter API key. Empty string disables generation.
            model: Model identifier on OpenRouter.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        # Token usage from the most recent successful call (for observability).
        self.last_usage: dict = {}

    def is_enabled(self) -> bool:
        """Return True when an API key is configured."""
        return bool(self.api_key)

    def generate(self, question: str, context_chunks: List[str]) -> str:
        """Generate an answer using the LLM with retrieved context.

        Args:
            question: User question in natural language.
            context_chunks: Retrieved text chunks to ground the answer.

        Returns:
            Generated answer string, or a fallback message on failure.
        """
        if not self.is_enabled():
            return "LLM not configured. Set OPENROUTER_API_KEY in .env to enable answer generation."

        context_text = "\n\n---\n\n".join(
            f"[Chunk {i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )

        try:
            resp = httpx.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "RAG Document QA",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(
                            context=context_text, question=question
                        )},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            self.last_usage = data.get("usage", {})
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as exc:
            logger.error("LLM API error %s: %s", exc.response.status_code, exc.response.text[:200])
            return f"LLM request failed (HTTP {exc.response.status_code}). Falling back to retrieved chunks."
        except httpx.RequestError as exc:
            logger.error("LLM network error: %s", exc)
            return "LLM request failed due to a network error. Falling back to retrieved chunks."
        except Exception as exc:
            logger.error("LLM unexpected error: %s", exc)
            return f"LLM error: {exc}"
