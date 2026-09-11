"""Unit tests for Escalation Decision Engine."""

import pytest
from src.escalation import EscalationEngine


@pytest.fixture
def escalation_engine():
    return EscalationEngine(
        intent_threshold=0.65,
        retrieval_threshold=0.60,
        grounding_threshold=0.70,
    )


def test_auto_handle_high_confidence(escalation_engine):
    evidence = [{"similarity_score": 0.85, "cleaned_brand_reply": "Reset your AirPods."}]
    res = escalation_engine.evaluate(
        customer_query="How do I reset my AirPods?",
        predicted_intent="connectivity_sync_bluetooth",
        intent_confidence=0.92,
        evidence_list=evidence,
        grounding_score=0.88,
        num_turns=1,
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert res["is_escalated"] is False
    assert res["risk_level"] == "LOW"


def test_escalate_on_critical_safety(escalation_engine):
    evidence = [{"similarity_score": 0.90, "cleaned_brand_reply": "Battery service."}]
    res = escalation_engine.evaluate(
        customer_query="My charger battery exploded and caught fire on my desk!!",
        predicted_intent="hardware_device_issue",
        intent_confidence=0.95,
        evidence_list=evidence,
        grounding_score=0.90,
        num_turns=1,
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert res["is_escalated"] is True
    assert res["risk_level"] == "HIGH"
    assert "battery exploded" in " ".join(res["reasons"])
    assert "Critical Hardware Safety" in res["recommended_queue"]


def test_escalate_on_legal_threat(escalation_engine):
    evidence = [{"similarity_score": 0.75, "cleaned_brand_reply": "Contact support."}]
    res = escalation_engine.evaluate(
        customer_query="I will hire a lawyer and sue your company for this refund!",
        predicted_intent="returns_refund_exchange",
        intent_confidence=0.80,
        evidence_list=evidence,
        grounding_score=0.80,
        num_turns=1,
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "Legal Escalations" in res["recommended_queue"]


def test_escalate_on_low_confidence(escalation_engine):
    evidence = [{"similarity_score": 0.50, "cleaned_brand_reply": "Unclear answer."}]
    res = escalation_engine.evaluate(
        customer_query="Something weird happened with my thing",
        predicted_intent="software_update_bug",
        intent_confidence=0.42,  # < 0.65 threshold
        evidence_list=evidence,
        grounding_score=0.55,   # < 0.70 threshold
        num_turns=1,
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert any("Low intent classification confidence" in r for r in res["reasons"])
