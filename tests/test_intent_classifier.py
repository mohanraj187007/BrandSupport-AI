"""Unit tests for Intent Classifiers (Majority, TF-IDF, and Semantic)."""

import pytest
from baselines.majority_baseline import MajorityClassBaseline
from baselines.tfidf_baseline import TFIDFIntentClassifier
from src.intent_classifier import SemanticIntentClassifier


@pytest.fixture
def sample_training_data():
    texts = [
        "iPhone battery drains fast and is swollen",
        "Screen cracked and not turning on",
        "iOS update froze my phone on boot logo",
        "Safari app crashing after update",
        "Apple ID locked unauthorized charge on bill",
        "Duplicate billing for iCloud subscription",
        "Package not delivered courier lost order",
        "Where is my shipping tracking number",
    ]
    labels = [
        "hardware_device_issue",
        "hardware_device_issue",
        "software_update_bug",
        "software_update_bug",
        "account_auth_billing",
        "account_auth_billing",
        "order_shipping_delivery",
        "order_shipping_delivery",
    ]
    return texts, labels


def test_majority_baseline(sample_training_data):
    texts, labels = sample_training_data
    model = MajorityClassBaseline()
    model.fit(texts, labels)

    preds = model.predict(["Any random query"])
    assert len(preds) == 1
    assert preds[0] in labels

    res = model.predict_single("test query")
    assert "predicted_intent" in res
    assert "confidence" in res
    assert res["model_type"] == "MajorityClassBaseline"


def test_tfidf_classifier(sample_training_data):
    texts, labels = sample_training_data
    model = TFIDFIntentClassifier()
    model.fit(texts, labels)

    res = model.predict_single("My iPhone battery won't hold a charge")
    assert "predicted_intent" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert sum(res["all_probabilities"].values()) == pytest.approx(1.0, rel=1e-2)


def test_semantic_classifier(sample_training_data):
    texts, labels = sample_training_data
    model = SemanticIntentClassifier()
    model.fit(texts, labels)

    res = model.predict_single("Where is my package delivery tracking?")
    assert "predicted_intent" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "margin" in res
    assert "entropy" in res
