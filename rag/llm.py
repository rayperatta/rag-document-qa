"""LLM answer generation module — OpenRouter API integration."""
import logging
import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer the user's question based ONLY on the provided context chunks. If the answer is not in the context, say "I couldn't find this information in the uploaded documents." Always cite which chunk(s) you used."""

USER_TEMPLATE = """Context chunks from uploaded documents:

{context}

---

Question: {question}

Answer based only on the above context. Be concise and cite sources."""


class LLMGenerator:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct:free"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def generate(self, question: str, context_chunks: list[str]) -> str:
        """Generate an answer using the LLM with retrieved context."""
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
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error {e.response.status_code}: {e.response.text[:200]}")
            return f"LLM request failed (HTTP {e.response.status_code}). Falling back to retrieved chunks."
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"LLM error: {e}"
