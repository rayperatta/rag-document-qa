"""Vector retrieval module — ChromaDB-backed semantic search."""
import logging
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class Retriever:
    """Manage document embeddings and semantic search via ChromaDB."""

    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        """Initialize the retriever with a persistent ChromaDB store.

        Args:
            persist_dir: Directory for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embed_fn = None
        self._init_store()

    def _init_store(self) -> None:
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
            logger.info("ChromaDB ready: %d chunks in collection", self._collection.count())
        except Exception as exc:
            logger.error("ChromaDB init failed: %s", exc)
            self._client = None

    def is_ready(self) -> bool:
        """Check whether the vector store contains any embeddings."""
        return self._client is not None and self._collection is not None and self._collection.count() > 0

    def add_documents(self, chunks: List[str], metadata: Dict) -> None:
        """Add text chunks to the vector store.

        Args:
            chunks: List of text strings to embed and store.
            metadata: Shared metadata dict (must include ``doc_id``).
        """
        if not self._collection:
            logger.warning("Cannot add documents: ChromaDB not initialized")
            return
        if not chunks:
            logger.debug("No chunks provided, skipping add_documents")
            return

        ids = [f"{metadata['doc_id']}_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk": i} for i in range(len(chunks))]

        self._collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        logger.debug("Added %d chunks for doc_id=%s", len(chunks), metadata.get("doc_id"))

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.

        Returns:
            List of dicts with ``content``, ``metadata``, and ``score`` keys.
        """
        results = self.search_with_ids(query, k=k)
        # Backwards-compatible shape (no IDs) for existing consumers.
        return [
            {"content": r["content"], "metadata": r["metadata"], "score": r["score"]}
            for r in results
        ]

    def search_with_ids(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks, including chunk IDs (for hybrid fusion).

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.

        Returns:
            List of dicts with ``id``, ``content``, ``metadata``, and ``score`` keys.
        """
        if not self.is_ready():
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []

        # Convert distance to similarity score (cosine distance → similarity)
        return [
            {
                "id": chunk_id,
                "content": doc,
                "metadata": meta,
                "score": 1 - dist,
            }
            for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists)
        ]

    def delete_by_metadata(self, key: str, value: str) -> None:
        """Delete all chunks matching a metadata key=value.

        Args:
            key: Metadata field name.
            value: Metadata field value to match.
        """
        if not self._collection:
            return

        results = self._collection.get(
            where={key: value},
            include=["metadatas"],
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info("Deleted %d chunks for %s=%s", len(results["ids"]), key, value)
