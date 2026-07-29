"""RAGAS evaluation harness for the RAG pipeline.

Answers the interview question: "how do you know your RAG works well?"
with numbers instead of vibes.

Usage:
    pip install -r requirements-dev.txt
    # .env needs OPENROUTER_API_KEY (LLM judge) and documents uploaded
    python -m eval.run_eval [--dataset eval/golden_dataset.json] [--k 5]

Metrics (RAGAS):
  - faithfulness       — is the answer grounded in the retrieved context? (anti-hallucination)
  - answer_relevancy   — does the answer address the question?
  - context_precision  — are the retrieved chunks relevant?
  - context_recall     — did we retrieve everything needed? (needs ground_truth)

The golden dataset is versioned JSON so evals are reproducible across
pipeline changes (chunking, reranking, prompts) — run before/after any
retrieval change and compare.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

DEFAULT_DATASET = Path(__file__).parent / "golden_dataset.json"


def build_components():
    """Wire the same components the API uses (chunker, retriever, hybrid, LLM)."""
    from rag.hybrid import HybridRetriever
    from rag.llm import LLMGenerator
    from rag.retriever import Retriever

    retriever = Retriever(
        persist_dir=str(BASE_DIR / "data" / "chroma"),
        collection_name="documents",
    )
    hybrid = HybridRetriever(
        retriever,
        reranker_enabled=os.getenv("RERANKER_ENABLED", "true").lower() == "true",
    )
    llm = LLMGenerator(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
    )
    return retriever, hybrid, llm


def run_pipeline(hybrid, llm, question: str, k: int):
    """Run one question through retrieval + generation; return RAGAS row."""
    results = hybrid.search(question, k=k)
    contexts = [r["content"] for r in results]
    answer = llm.generate(question, contexts)
    return answer, contexts


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Golden dataset JSON")
    parser.add_argument("--k", type=int, default=5, help="top-k retrieval")
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY required (LLM-as-judge + answer generation).")
        return 1

    golden = json.loads(Path(args.dataset).read_text())
    _, hybrid, llm = build_components()

    if not hybrid.vector.is_ready():
        print("ERROR: no documents indexed. Upload PDFs via the API first.")
        return 1

    rows = []
    for item in golden:
        answer, contexts = run_pipeline(hybrid, llm, item["question"], args.k)
        rows.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item.get("ground_truth", ""),
            }
        )
        print(f"  ✓ {item['question'][:70]}")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-4o-mini"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    )

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
    )

    print("\n=== RAGAS results ===")
    print(result)
    out = Path("eval/results.json")
    out.write_text(json.dumps(result.to_pandas().to_dict(orient="records"), indent=2))
    print(f"\nPer-question scores written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
