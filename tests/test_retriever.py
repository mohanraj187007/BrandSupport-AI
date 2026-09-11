"""Unit tests for FAISS Retriever and Anti-Leakage partitioning."""

import pytest
from src.retriever import FAISSRetriever


@pytest.fixture
def sample_corpus():
    return [
        {
            "conversation_id": "conv_001",
            "brand_name": "AppleSupport",
            "customer_query": "@AppleSupport battery health is 75%",
            "cleaned_customer_query": "battery health is 75%",
            "brand_reply": "@cust Get a battery replacement at Apple Store",
            "cleaned_brand_reply": "Get a battery replacement at Apple Store",
            "intent": "hardware_device_issue",
            "num_turns": 2,
        },
        {
            "conversation_id": "conv_002",
            "brand_name": "AppleSupport",
            "customer_query": "@AppleSupport how to backup to iCloud",
            "cleaned_customer_query": "how to backup to iCloud",
            "brand_reply": "@cust Go to Settings > iCloud > Backup",
            "cleaned_brand_reply": "Go to Settings > iCloud > Backup",
            "intent": "feature_how_to_inquiry",
            "num_turns": 2,
        },
        {
            "conversation_id": "conv_gold_eval",
            "brand_name": "AppleSupport",
            "customer_query": "@AppleSupport evaluation test query",
            "cleaned_customer_query": "evaluation test query",
            "brand_reply": "@cust test reply",
            "cleaned_brand_reply": "test reply",
            "intent": "general_feedback_complaint",
            "num_turns": 2,
        }
    ]


def test_retriever_build_and_query(sample_corpus):
    retriever = FAISSRetriever()
    # Build index excluding evaluation items (Anti-Leakage)
    retriever.build_index(sample_corpus, exclude_conversation_ids={"conv_gold_eval"})

    assert retriever.index is not None
    assert retriever.index.ntotal == 2

    # Query similarity
    results = retriever.retrieve("my battery is dying very quickly", top_k=2)
    assert len(results) >= 1
    assert results[0]["intent"] == "hardware_device_issue"
    assert "conv_gold_eval" not in [r["conversation_id"] for r in results]
    assert 0.0 <= results[0]["similarity_score"] <= 1.0


def test_retriever_anti_leakage_filter(sample_corpus):
    retriever = FAISSRetriever()
    retriever.build_index(sample_corpus)
    # Dynamically exclude conv_001
    results = retriever.retrieve("battery health", top_k=5, exclude_conversation_ids={"conv_001"})
    retrieved_ids = [r["conversation_id"] for r in results]
    assert "conv_001" not in retrieved_ids
