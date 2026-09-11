"""Escalation Decision Engine for determining AUTO_HANDLE vs ESCALATE_TO_HUMAN."""

import logging
from typing import Dict, Any, List, Optional
from src.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    GROUNDING_THRESHOLD,
    HIGH_URGENCY_KEYWORDS,
    ESCALATION_QUEUES,
)
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


class EscalationEngine:
    """
    Deterministic & rule-calibrated escalation engine that decides whether
    a customer query can be safely auto-handled or must be routed to human agents.
    """

    def __init__(
        self,
        intent_threshold: float = INTENT_CONFIDENCE_THRESHOLD,
        retrieval_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
        grounding_threshold: float = GROUNDING_THRESHOLD,
    ):
        self.intent_threshold = intent_threshold
        self.retrieval_threshold = retrieval_threshold
        self.grounding_threshold = grounding_threshold

    def evaluate(
        self,
        customer_query: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_list: List[Dict[str, Any]],
        grounding_score: float,
        num_turns: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluates query signals and makes the final routing decision.
        
        Returns:
            Dict with 'decision', 'reasons', 'recommended_queue', 'risk_level', 'aggregate_confidence'.
        """
        triggers: List[str] = []
        cleaned_query = clean_text(customer_query).lower()

        # 1. Critical Urgency / Safety / Legal / Fraud Keywords Check
        detected_keywords = []
        for kw in HIGH_URGENCY_KEYWORDS:
            if kw in cleaned_query:
                detected_keywords.append(kw)

        if detected_keywords:
            triggers.append(f"Critical safety/legal/fraud keywords detected: {', '.join(detected_keywords)}")

        # 2. Intent Confidence Threshold Check
        if intent_confidence < self.intent_threshold:
            triggers.append(
                f"Low intent classification confidence ({intent_confidence:.2f} < {self.intent_threshold:.2f})"
            )

        # 3. Retrieval Similarity & Evidence Quality Check
        if not evidence_list:
            triggers.append("No relevant historical customer support evidence found in database")
        else:
            top_sim = evidence_list[0].get("similarity_score", 0.0)
            if top_sim < self.retrieval_threshold:
                triggers.append(
                    f"Top retrieved evidence similarity too low ({top_sim:.2f} < {self.retrieval_threshold:.2f})"
                )

        # 4. Reply Grounding Faithfulness Check
        if grounding_score < self.grounding_threshold:
            triggers.append(
                f"Generated reply grounding score below threshold ({grounding_score:.2f} < {self.grounding_threshold:.2f})"
            )

        # 5. Multi-turn unresolved escalation
        if num_turns >= 3:
            triggers.append(f"Prolonged multi-turn dialogue ({num_turns} turns) indicates unresolved friction")

        # Final Decision
        if triggers:
            decision = "ESCALATE_TO_HUMAN"
            risk_level = "HIGH" if detected_keywords else "MEDIUM"
        else:
            decision = "AUTO_HANDLE"
            risk_level = "LOW"

        # Routing queue
        recommended_queue = ESCALATION_QUEUES.get(predicted_intent, "General Support Specialist")
        if detected_keywords:
            if any(k in ["fraud", "stolen", "unauthorized", "hacked"] for k in detected_keywords):
                recommended_queue = "Security, Fraud & Billing Escalation Team"
            elif any(k in ["lawyer", "attorney", "sue", "court", "lawsuit", "bbb"] for k in detected_keywords):
                recommended_queue = "Executive Customer Relations & Legal Escalations"
            elif any(k in ["safety hazard", "battery exploded", "smoke", "burned", "fire"] for k in detected_keywords):
                recommended_queue = "Critical Hardware Safety Incident Response Unit"

        # Aggregate confidence score
        top_sim = evidence_list[0].get("similarity_score", 0.0) if evidence_list else 0.0
        aggregate_confidence = round(
            0.4 * intent_confidence + 0.35 * top_sim + 0.25 * grounding_score, 4
        )

        return {
            "decision": decision,
            "reasons": triggers if triggers else ["High confidence prediction, strong evidence match, and high grounding score."],
            "is_escalated": decision == "ESCALATE_TO_HUMAN",
            "recommended_queue": recommended_queue,
            "risk_level": risk_level,
            "aggregate_confidence": aggregate_confidence,
            "metrics_breakdown": {
                "intent_confidence": intent_confidence,
                "retrieval_top_similarity": top_sim,
                "grounding_score": grounding_score,
            }
        }
