"""Vector retrieval module — ChromaDB-backed semantic search."""
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embed_fn = None
        self._init_store()

    def _init_store(self):
        """Initialize ChromaDB with local sentence-transformers embeddings."""
        try:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB ready: {self._collection.count()} chunks in collection")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            self._client = None

    def is_ready(self) -> bool:
        return self._client is not None and self._collection is not None and self._collection.count() > 0

    def add_documents(self, chunks: list[str], metadata: dict):
        """Add text chunks to the vector store."""
        if not self._collection:
            return

        ids = [f"{metadata['doc_id']}_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk": i} for i in range(len(chunks))]

        self._collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for relevant chunks. Returns list of {content, metadata, score}."""
        if not self.is_ready():
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []

        # Convert distance to similarity score (cosine distance → similarity)
        return [
            {
                "content": doc,
                "metadata": meta,
                "score": 1 - dist,
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def delete_by_metadata(self, key: str, value: str):
        """Delete all chunks matching a metadata key=value."""
        if not self._collection:
            return

        results = self._collection.get(
            where={key: value},
            include=["metadatas"],
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for {key}={value}")
