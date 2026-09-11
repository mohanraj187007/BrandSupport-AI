"""Semantic Intent Classifier using Sentence Transformer Embeddings and Calibrated Classification."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.config import EMBEDDING_MODEL_NAME, SEMANTIC_CLASSIFIER_PATH
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


class SemanticIntentClassifier:
    """
    Advanced semantic intent classifier leveraging dense sentence embeddings
    coupled with Platt-calibrated multi-class classification.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.encoder = None
        self.classifier = None
        self.classes_: List[str] = []
        self._init_encoder()

    def _init_encoder(self):
        """Initializes the SentenceTransformer encoder with graceful fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_name}...")
            self.encoder = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"SentenceTransformer not available or failed to load ({e}). Using dense vectorizer fallback.")
            self.encoder = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts into normalized semantic embedding vectors."""
        cleaned_texts = [clean_text(t) for t in texts]
        if self.encoder is not None:
            embeddings = self.encoder.encode(cleaned_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.astype("float32")
        else:
            # Fallback lightweight deterministic dense character-word embedding
            from sklearn.feature_extraction.text import TfidfVectorizer
            if not hasattr(self, "_fallback_vec"):
                self._fallback_vec = TfidfVectorizer(max_features=384, sublinear_tf=True)
                emb = self._fallback_vec.fit_transform(cleaned_texts).toarray()
            else:
                emb = self._fallback_vec.transform(cleaned_texts).toarray()
            # L2 normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (emb / norms).astype("float32")

    def fit(self, texts: List[str], labels: List[str]) -> "SemanticIntentClassifier":
        """Trains the calibrated semantic classifier."""
        if not texts or not labels:
            raise ValueError("Training texts and labels cannot be empty.")

        logger.info(f"Encoding {len(texts)} samples for semantic intent training...")
        embeddings = self.encode(texts)

        base_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
        
        # Calibrated classifier for robust confidence estimation
        if len(texts) >= 10:
            self.classifier = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        else:
            self.classifier = base_clf

        self.classifier.fit(embeddings, labels)
        self.classes_ = list(self.classifier.classes_)
        logger.info(f"Trained SemanticIntentClassifier across {len(self.classes_)} classes.")
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts intent labels for given texts."""
        if self.classifier is None:
            raise ValueError("Classifier is not fitted.")
        embeddings = self.encode(texts)
        return list(self.classifier.predict(embeddings))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predicts probability distributions."""
        if self.classifier is None:
            raise ValueError("Classifier is not fitted.")
        embeddings = self.encode(texts)
        return self.classifier.predict_proba(embeddings)

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Runs single-query inference with confidence calibration and margin estimation."""
        if self.classifier is None:
            raise ValueError("Classifier is not fitted.")

        probs = self.predict_proba([text])[0]
        max_idx = int(np.argmax(probs))
        predicted_class = self.classes_[max_idx]
        confidence = float(probs[max_idx])

        # Calculate prediction margin & entropy
        sorted_probs = sorted(probs, reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
        
        # Normalized entropy (0 = totally certain, 1 = uniform distribution)
        safe_probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(safe_probs * np.log2(safe_probs)) / np.log2(len(self.classes_))

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes_, probs)}

        return {
            "predicted_intent": predicted_class,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "entropy": round(float(entropy), 4),
            "all_probabilities": prob_dict,
            "model_type": "Semantic_SentenceTransformer_Calibrated",
        }

    def save(self, filepath: Path = SEMANTIC_CLASSIFIER_PATH) -> None:
        """Saves model state and metadata."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # We store model weights, classes, and fallback vectorizer
        data = {
            "model_name": self.model_name,
            "classifier": self.classifier,
            "classes_": self.classes_,
            "fallback_vec": getattr(self, "_fallback_vec", None),
        }
        joblib.dump(data, filepath)
        logger.info(f"Saved SemanticIntentClassifier artifact to {filepath}")

    @classmethod
    def load(cls, filepath: Path = SEMANTIC_CLASSIFIER_PATH) -> "SemanticIntentClassifier":
        """Loads semantic classifier artifact."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        data = joblib.load(filepath)
        instance = cls(model_name=data.get("model_name", EMBEDDING_MODEL_NAME))
        instance.classifier = data["classifier"]
        instance.classes_ = data["classes_"]
        if data.get("fallback_vec") is not None:
            instance._fallback_vec = data["fallback_vec"]
        logger.info(f"Loaded SemanticIntentClassifier with classes {instance.classes_}")
        return instance
