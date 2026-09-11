"""FAISS Semantic Retrieval Engine for historical customer support conversations."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import numpy as np

from src.config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """
    Historical conversation retrieval engine utilizing FAISS Inner-Product (cosine) search
    with metadata linking and evaluation anti-leakage filters.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._init_encoder()

    def _init_encoder(self):
        """Initializes the SentenceTransformer encoder."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer in retriever: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer in retriever ({e}). Using dense vectorizer fallback.")
            self.encoder = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts into normalized float32 embedding vectors."""
        cleaned_texts = [clean_text(t) for t in texts]
        if self.encoder is not None:
            emb = self.encoder.encode(cleaned_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return emb.astype("float32")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            if not hasattr(self, "_fallback_vec"):
                self._fallback_vec = TfidfVectorizer(max_features=384, sublinear_tf=True)
                emb = self._fallback_vec.fit_transform(cleaned_texts).toarray()
            else:
                emb = self._fallback_vec.transform(cleaned_texts).toarray()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (emb / norms).astype("float32")

    def build_index(
        self,
        conversations: List[Dict[str, Any]],
        exclude_conversation_ids: Optional[Set[str]] = None,
    ) -> "FAISSRetriever":
        """
        Builds FAISS index over historical customer queries while enforcing anti-leakage filters.
        
        Args:
            conversations: Reconstructed conversation dictionaries.
            exclude_conversation_ids: Conversation IDs to exclude (e.g. golden evaluation set).
        """
        import faiss

        if exclude_conversation_ids is None:
            exclude_conversation_ids = set()

        filtered_convs = [
            c for c in conversations
            if c.get("conversation_id") not in exclude_conversation_ids
            and c.get("cleaned_customer_query")
            and c.get("cleaned_brand_reply")
        ]

        if not filtered_convs:
            raise ValueError("No conversations available to build retrieval index after filtering.")

        logger.info(f"Building retrieval index on {len(filtered_convs)} historical conversations (excluded {len(exclude_conversation_ids)} evaluation items).")

        corpus_texts = [c["cleaned_customer_query"] for c in filtered_convs]
        embeddings = self.encode(corpus_texts)
        dimension = embeddings.shape[1]

        # FAISS IndexFlatIP computes exact inner product (cosine similarity on L2-normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = filtered_convs

        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors of dimension {dimension}.")
        return self

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        brand_name: Optional[str] = None,
        exclude_conversation_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar historical conversations and approved resolutions.
        
        Args:
            query: Input customer query string.
            top_k: Number of historical records to retrieve.
            brand_name: Optional brand filter.
            exclude_conversation_ids: Set of conversation IDs to dynamically exclude.
            
        Returns:
            List of retrieved historical evidence items with similarity scores.
        """
        if self.index is None or not self.metadata:
            logger.warning("Retrieval index is empty.")
            return []

        cleaned_query = clean_text(query)
        query_vec = self.encode([cleaned_query])

        # Fetch extra candidates to allow brand & anti-leakage filtering
        fetch_k = min(top_k * 5, len(self.metadata))
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        exclude_set = exclude_conversation_ids or set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            conv_id = meta.get("conversation_id")

            if conv_id in exclude_set:
                continue

            if brand_name and meta.get("brand_name", "").lower() != brand_name.lower():
                continue

            # Normalized cosine similarity score in [0.0, 1.0]
            sim_score = max(0.0, min(1.0, float(score)))

            results.append({
                "conversation_id": conv_id,
                "brand_name": meta.get("brand_name"),
                "similarity_score": round(sim_score, 4),
                "intent": meta.get("intent"),
                "customer_query": meta.get("customer_query"),
                "cleaned_customer_query": meta.get("cleaned_customer_query"),
                "brand_reply": meta.get("brand_reply"),
                "cleaned_brand_reply": meta.get("cleaned_brand_reply"),
                "num_turns": meta.get("num_turns", 1),
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: Path = FAISS_INDEX_PATH) -> None:
        """Persists FAISS index binary and metadata JSON to disk."""
        import faiss
        filepath.parent.mkdir(parents=True, exist_ok=True)
        meta_path = filepath.with_suffix(".meta.json")

        if self.index is not None:
            faiss.write_index(self.index, str(filepath))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_name": self.model_name,
                    "metadata": self.metadata,
                }, f, indent=2)
            logger.info(f"Saved FAISS index to {filepath} and metadata to {meta_path}")

    @classmethod
    def load(cls, filepath: Path = FAISS_INDEX_PATH) -> "FAISSRetriever":
        """Loads FAISS index and metadata from disk."""
        import faiss
        meta_path = filepath.with_suffix(".meta.json")
        if not filepath.exists() or not meta_path.exists():
            raise FileNotFoundError(f"FAISS index or metadata missing at {filepath}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        instance = cls(model_name=meta_data.get("model_name", EMBEDDING_MODEL_NAME))
        instance.index = faiss.read_index(str(filepath))
        instance.metadata = meta_data.get("metadata", [])
        logger.info(f"Loaded FAISS index with {instance.index.ntotal} items from {filepath}")
        return instance
