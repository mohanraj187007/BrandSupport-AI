"""Majority Class Baseline Classifier."""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np


class MajorityClassBaseline:
    """
    A baseline classifier that always predicts the most frequent class in the training distribution.
    """

    def __init__(self):
        self.majority_class: Optional[str] = None
        self.class_priors: Dict[str, float] = {}
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]) -> "MajorityClassBaseline":
        """Fits the majority class model by computing label frequencies."""
        if not labels:
            raise ValueError("Training labels cannot be empty.")

        counts = Counter(labels)
        total = len(labels)
        self.majority_class = counts.most_common(1)[0][0]
        self.classes_ = sorted(list(counts.keys()))
        self.class_priors = {cls: counts[cls] / total for cls in self.classes_}
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts the majority class for all input texts."""
        if self.majority_class is None:
            raise ValueError("Model has not been fitted yet.")
        return [self.majority_class] * len(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns probability distribution matching training priors for each sample."""
        if not self.classes_:
            raise ValueError("Model has not been fitted yet.")
        probs = np.array([self.class_priors.get(cls, 0.0) for cls in self.classes_])
        return np.tile(probs, (len(texts), 1))

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Inference for a single text string."""
        probs = self.predict_proba([text])[0]
        max_idx = int(np.argmax(probs))
        return {
            "predicted_intent": self.classes_[max_idx],
            "confidence": float(probs[max_idx]),
            "all_probabilities": {cls: float(p) for cls, p in zip(self.classes_, probs)},
            "model_type": "MajorityClassBaseline",
        }
