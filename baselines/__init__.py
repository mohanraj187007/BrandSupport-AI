"""Baseline Intent Classifiers."""
from baselines.majority_baseline import MajorityClassBaseline
from baselines.tfidf_baseline import TFIDFIntentClassifier

__all__ = ["MajorityClassBaseline", "TFIDFIntentClassifier"]
