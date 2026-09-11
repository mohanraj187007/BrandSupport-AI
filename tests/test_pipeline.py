"""Integration tests for the unified BrandSupport AI Pipeline."""

import pytest
from src.pipeline import BrandSupportPipeline
from src.intent_classifier import SemanticIntentClassifier
from src.retriever import FAISSRetriever
from src.reply_generator import GroundedReplyGenerator
from src.escalation import EscalationEngine


@pytest.fixture
def mock_pipeline():
    # Setup trained small pipeline
    convs = [
        {
            "conversation_id": "c1",
            "brand_name": "AppleSupport",
            "customer_query": "@AppleSupport battery health 72% drains fast",
            "cleaned_customer_query": "battery health 72% drains fast",
            "brand_reply": "@cust Book a battery replacement at Apple Store",
            "cleaned_brand_reply": "Book a battery replacement at Apple Store",
            "intent": "hardware_device_issue",
            "num_turns": 2,
        },
        {
            "conversation_id": "c2",
            "brand_name": "AppleSupport",
            "customer_query": "@AppleSupport how to setup Family Sharing",
            "cleaned_customer_query": "how to setup Family Sharing",
            "brand_reply": "@cust Go to Settings > Your Name > Family Sharing",
            "cleaned_brand_reply": "Go to Settings > Your Name > Family Sharing",
            "intent": "feature_how_to_inquiry",
            "num_turns": 2,
        }
    ]
    texts = [c["cleaned_customer_query"] for c in convs]
    labels = [c["intent"] for c in convs]

    clf = SemanticIntentClassifier()
    clf.fit(texts, labels)

    retriever = FAISSRetriever()
    retriever.build_index(convs)

    reply_gen = GroundedReplyGenerator()
    escalation = EscalationEngine()

    return BrandSupportPipeline(
        brand_name="AppleSupport",
        intent_classifier=clf,
        retriever=retriever,
        reply_generator=reply_gen,
        escalation_engine=escalation,
    )


def test_pipeline_end_to_end_auto_handle(mock_pipeline):
    query = "My battery is dying in 2 hours and maximum capacity is 70%"
    res = mock_pipeline.process_query(query)

    assert "predicted_intent" in res
    assert "intent_confidence" in res
    assert "retrieved_evidence" in res
    assert "generated_reply" in res
    assert "decision" in res
    assert res["decision"] in ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
    assert "recommended_queue" in res
    assert "grounding_score" in res


def test_pipeline_end_to_end_escalation(mock_pipeline):
    query = "My battery exploded in my hands and burned my table! I will sue your company!"
    res = mock_pipeline.process_query(query)

    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert res["is_escalated"] is True
    assert res["risk_level"] == "HIGH"
    assert len(res["escalation_reasons"]) > 0
