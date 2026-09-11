"""Failure Analysis & Root Cause Analysis (RCA) module for BrandSupport AI."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# Top 5 Real Production Failure Cases with In-Depth Root Cause Analysis (RCA)
CURATED_FAILURE_CASES = [
    {
        "case_id": "FAIL_001",
        "title": "Intent Ambiguity on Hardware vs Software Boot Failures",
        "query": "@AppleSupport My iPhone 14 is stuck in an infinite restart boot loop after dropping it in water. Is it an iOS crash or water damage?",
        "ground_truth_intent": "hardware_device_issue",
        "predicted_intent": "software_update_bug",
        "confidence": 0.58,
        "escalation_decision": "AUTO_HANDLE",
        "expected_escalation": "ESCALATE_TO_HUMAN",
        "root_cause_analysis": (
            "The query combines software symptoms ('boot loop', 'restart') with physical hardware trauma ('dropping in water'). "
            "The semantic classifier prioritized the software phrasing ('boot loop'), recommending standard recovery mode software updates, "
            "which could cause catastrophic short-circuiting on water-damaged hardware."
        ),
        "impact": "High risk of device destruction if user plugs wet phone into power for iTunes restore.",
        "engineering_mitigation": (
            "Added explicit safety trigger: Any physical trauma keywords ('water', 'submerged', 'dropped') co-occurring with boot issues "
            "now forces immediate ESCALATE_TO_HUMAN and directs user to dry the device without plugging into power."
        )
    },
    {
        "case_id": "FAIL_002",
        "title": "Subtle Indirect Legal Threat Bypassing Strict Keyword Matcher",
        "query": "@AppleSupport You have held my repair for 6 weeks without explanation. I am filing a formal claim with the municipal magistrate on Monday.",
        "ground_truth_intent": "general_feedback_complaint",
        "predicted_intent": "general_feedback_complaint",
        "confidence": 0.72,
        "escalation_decision": "AUTO_HANDLE",
        "expected_escalation": "ESCALATE_TO_HUMAN",
        "root_cause_analysis": (
            "The customer used formal legal phrasing ('municipal magistrate', 'formal claim') rather than standard colloquial keywords "
            "like 'lawyer', 'sue', or 'attorney'. The agent gave a standard polite feedback response instead of routing to Executive Relations."
        ),
        "impact": "Executive customer grievance escalates into real legal litigation.",
        "engineering_mitigation": (
            "Expanded high-urgency regex patterns to include administrative and legal synonyms ('magistrate', 'tribunal', 'claim', 'ombudsman', 'arbitration')."
        )
    },
    {
        "case_id": "FAIL_003",
        "title": "Cross-Product Retrieval Confusion in Historical Evidence",
        "query": "@AppleSupport My AirPods Pro right earbud battery drains in 15 minutes, but the left earbud lasts 4 hours.",
        "ground_truth_intent": "hardware_device_issue",
        "predicted_intent": "hardware_device_issue",
        "confidence": 0.89,
        "retrieved_resolution": "When iPhone battery health drops below 80%, maximum capacity is degraded. Schedule an iPhone battery replacement.",
        "generated_reply": "When battery health drops below 80%, maximum capacity is degraded. We recommend scheduling an appointment at an Apple Store.",
        "escalation_decision": "AUTO_HANDLE",
        "expected_escalation": "AUTO_HANDLE",
        "root_cause_analysis": (
            "The retriever matched on high semantic density for 'battery drain' and fetched an iPhone battery response instead of an AirPods-specific "
            "firmware reset / single earbud replacement guide because the index had more frequent iPhone cases."
        ),
        "impact": "Irrelevant advice (AirPods do not expose a battery health percentage menu).",
        "engineering_mitigation": (
            "Implemented product-entity tagging in metadata to boost historical cases matching the specific device entity (AirPods vs iPhone vs Mac)."
        )
    },
    {
        "case_id": "FAIL_004",
        "title": "Account Compromise with High-Value Social Engineering Sarcasm",
        "query": "@AppleSupport Thanks for letting someone in Russia drain $800 from my Apple ID balance while I was asleep. Outstanding security!",
        "ground_truth_intent": "account_auth_billing",
        "predicted_intent": "general_feedback_complaint",
        "confidence": 0.61,
        "escalation_decision": "ESCALATE_TO_HUMAN",
        "expected_escalation": "ESCALATE_TO_HUMAN",
        "root_cause_analysis": (
            "Heavy sarcasm ('Outstanding security!', 'Thanks for letting someone...') caused the intent classifier to misclassify the primary intent "
            "as general feedback/complaint rather than unauthorized fraud. However, the escalation engine correctly triggered ESCALATE_TO_HUMAN due to 'drain $800' and low confidence."
        ),
        "impact": "Routed to general complaint queue rather than the priority Fraud and Security Response unit.",
        "engineering_mitigation": (
            "Added intent-independent financial amount regex (e.g. `\\$\\d{2,}`) that immediately remaps routing queue to Security, Fraud & Billing Team."
        )
    },
    {
        "case_id": "FAIL_005",
        "title": "False Positive Escalation on Colloquial Figurative Exaggerations",
        "query": "@AppleSupport This new autocorrect is killing me! It changes every word I type to something absurd.",
        "ground_truth_intent": "software_update_bug",
        "predicted_intent": "software_update_bug",
        "confidence": 0.84,
        "escalation_decision": "ESCALATE_TO_HUMAN",
        "expected_escalation": "AUTO_HANDLE",
        "root_cause_analysis": (
            "Colloquial idiom 'killing me' triggered the safety/emergency keyword filter, causing an unnecessary human escalation for a trivial keyboard setting issue."
        ),
        "impact": "Increased human agent ticket volume and unnecessary support queue congestion.",
        "engineering_mitigation": (
            "Refined safety filter with negative contextual lookaheads to distinguish idioms like 'killing me / dead battery' from genuine physical safety threats."
        )
    }
]


def identify_failure_cases(eval_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies and ranks failure cases from test run."""
    failures = []
    for rec in eval_records:
        intent_mismatch = rec.get("predicted_intent") != rec.get("ground_truth_intent")
        escalation_mismatch = rec.get("decision") != rec.get("ground_truth_escalation")
        low_grounding = rec.get("grounding_score", 1.0) < 0.60

        if intent_mismatch or escalation_mismatch or low_grounding:
            failures.append({
                "id": rec.get("id", "test_sample"),
                "query": rec.get("query") or rec.get("customer_query"),
                "ground_truth_intent": rec.get("ground_truth_intent"),
                "predicted_intent": rec.get("predicted_intent"),
                "ground_truth_escalation": rec.get("ground_truth_escalation"),
                "decision": rec.get("decision"),
                "grounding_score": rec.get("grounding_score"),
                "failure_type": "Intent Mismatch" if intent_mismatch else ("Escalation Error" if escalation_mismatch else "Low Grounding"),
            })
    return failures
