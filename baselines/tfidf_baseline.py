"""TF-IDF + Logistic Regression Baseline Intent Classifier."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


class TFIDFIntentClassifier:
    """
    Baseline classifier combining Sublinear TF-IDF n-gram vectorizer with balanced Logistic Regression.
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2), c_param: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
        )
        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier),
        ])
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]) -> "TFIDFIntentClassifier":
        """Trains the TF-IDF and Logistic Regression pipeline."""
        if not texts or not labels:
            raise ValueError("Training texts and labels cannot be empty.")

        cleaned_texts = [clean_text(t) for t in texts]
        self.pipeline.fit(cleaned_texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        logger.info(f"Fitted TF-IDF Intent Classifier with {len(self.classes_)} classes.")
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts the intent classes for input texts."""
        cleaned_texts = [clean_text(t) for t in texts]
        return list(self.pipeline.predict(cleaned_texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predicts class probabilities."""
        cleaned_texts = [clean_text(t) for t in texts]
        return self.pipeline.predict_proba(cleaned_texts)

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Runs single-text inference and returns prediction, calibrated confidence and class probabilities."""
        cleaned = clean_text(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        max_idx = int(np.argmax(probs))
        predicted_class = self.classes_[max_idx]
        confidence = float(probs[max_idx])

        # Compute margin between top 1 and top 2 predictions for decision reliability
        sorted_probs = sorted(probs, reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])

        prob_dict = {cls: float(p) for cls, p in zip(self.classes_, probs)}

        return {
            "predicted_intent": predicted_class,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "all_probabilities": prob_dict,
            "model_type": "TFIDF_LogisticRegression",
        }

    def save(self, filepath: Path) -> None:
        """Persists trained model artifact to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved TFIDFIntentClassifier to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "TFIDFIntentClassifier":
        """Loads model artifact from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        model = joblib.load(filepath)
        logger.info(f"Loaded TFIDFIntentClassifier from {filepath}")
        return model
