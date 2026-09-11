"""Evaluation Suite for BrandSupport AI."""
from evaluation.metrics import (
    compute_intent_metrics,
    compute_retrieval_metrics,
    compute_escalation_metrics,
    compute_reply_text_metrics,
)
from evaluation.llm_judge import LLMJudge
from evaluation.failure_analyzer import identify_failure_cases, CURATED_FAILURE_CASES
from evaluation.evaluator import run_full_evaluation

__all__ = [
    "compute_intent_metrics",
    "compute_retrieval_metrics",
    "compute_escalation_metrics",
    "compute_reply_text_metrics",
    "LLMJudge",
    "identify_failure_cases",
    "CURATED_FAILURE_CASES",
    "run_full_evaluation",
]
