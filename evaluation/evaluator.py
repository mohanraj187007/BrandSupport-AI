"""Full benchmark evaluation runner comparing Majority baseline, TF-IDF baseline, and Semantic Pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.config import (
    GOLDEN_SET_PATH,
    EVAL_RESULTS_PATH,
    DEFAULT_BRAND,
    INTENT_LABELS,
    SEMANTIC_CLASSIFIER_PATH,
    FAISS_INDEX_PATH,
    TFIDF_MODEL_PATH,
)
from baselines.majority_baseline import MajorityClassBaseline
from baselines.tfidf_baseline import TFIDFIntentClassifier
from src.intent_classifier import SemanticIntentClassifier
from src.retriever import FAISSRetriever
from src.reply_generator import GroundedReplyGenerator
from src.escalation import EscalationEngine
from src.pipeline import BrandSupportPipeline
from evaluation.metrics import (
    compute_intent_metrics,
    compute_retrieval_metrics,
    compute_escalation_metrics,
    compute_reply_text_metrics,
)
from evaluation.llm_judge import LLMJudge
from evaluation.failure_analyzer import identify_failure_cases, CURATED_FAILURE_CASES

logger = logging.getLogger(__name__)


def run_full_evaluation(
    golden_set_path: Path = GOLDEN_SET_PATH,
    brand_name: str = DEFAULT_BRAND,
    output_path: Path = EVAL_RESULTS_PATH,
) -> Dict[str, Any]:
    """
    Executes comprehensive comparative evaluation across Majority Baseline, TF-IDF Baseline,
    and the full Semantic BrandSupport AI Pipeline against the human-annotated golden set.
    """
    if not golden_set_path.exists():
        from src.dataset_generator import generate_golden_evaluation_set
        generate_golden_evaluation_set(golden_set_path)

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_records = json.load(f)

    logger.info(f"Loaded {len(golden_records)} golden test records for evaluation.")

    queries = [r["query"] for r in golden_records]
    y_true_intent = [r["ground_truth_intent"] for r in golden_records]
    y_true_escalation = [r["ground_truth_escalation"] for r in golden_records]
    target_replies = [r.get("ground_truth_target_reply", "") for r in golden_records]

    # 1. Evaluate Baseline 1: Majority Class Baseline
    majority_model = MajorityClassBaseline()
    majority_model.fit(queries, y_true_intent)
    maj_preds = majority_model.predict(queries)
    majority_metrics = compute_intent_metrics(y_true_intent, maj_preds, INTENT_LABELS)

    # 2. Evaluate Baseline 2: TF-IDF + Logistic Regression
    if TFIDF_MODEL_PATH.exists():
        tfidf_model = TFIDFIntentClassifier.load(TFIDF_MODEL_PATH)
    else:
        tfidf_model = TFIDFIntentClassifier()
        tfidf_model.fit(queries, y_true_intent)
        tfidf_model.save(TFIDF_MODEL_PATH)

    tfidf_preds = tfidf_model.predict(queries)
    tfidf_metrics = compute_intent_metrics(y_true_intent, tfidf_preds, INTENT_LABELS)

    # 3. Evaluate Semantic Pipeline (Primary Model)
    pipeline = BrandSupportPipeline.load_from_artifacts(
        brand_name=brand_name,
        classifier_path=SEMANTIC_CLASSIFIER_PATH,
        index_path=FAISS_INDEX_PATH,
    )

    primary_preds = []
    escalation_preds = []
    retrieved_all = []
    generated_replies = []
    grounding_scores = []
    eval_detailed_records = []

    for rec in golden_records:
        q = rec["query"]
        res = pipeline.process_query(q, brand_name=brand_name)
        
        primary_preds.append(res["predicted_intent"])
        escalation_preds.append(res["decision"])
        retrieved_all.append(res["retrieved_evidence"])
        generated_replies.append(res["generated_reply"])
        grounding_scores.append(res["grounding_score"])

        eval_detailed_records.append({
            "id": rec.get("id"),
            "customer_query": q,
            "ground_truth_intent": rec["ground_truth_intent"],
            "predicted_intent": res["predicted_intent"],
            "ground_truth_escalation": rec["ground_truth_escalation"],
            "decision": res["decision"],
            "intent_confidence": res["intent_confidence"],
            "grounding_score": res["grounding_score"],
            "generated_reply": res["generated_reply"],
            "target_reply": rec.get("ground_truth_target_reply", ""),
            "retrieved_evidence": res["retrieved_evidence"],
            "escalation_reasons": res["escalation_reasons"],
            "recommended_queue": res["recommended_queue"],
        })

    primary_intent_metrics = compute_intent_metrics(y_true_intent, primary_preds, INTENT_LABELS)
    retrieval_metrics = compute_retrieval_metrics(retrieved_all, y_true_intent)
    escalation_metrics = compute_escalation_metrics(y_true_escalation, escalation_preds)
    reply_text_metrics = compute_reply_text_metrics(generated_replies, target_replies)

    # 4. LLM-as-a-Judge Evaluation
    judge = LLMJudge()
    judge_results = judge.evaluate_batch(eval_detailed_records)

    # 5. Failure Analysis
    failures = identify_failure_cases(eval_detailed_records)

    # Consolidated Results Object
    benchmark_report = {
        "dataset_size": len(golden_records),
        "brand": brand_name,
        "models_comparison": {
            "Majority_Class_Baseline": {
                "accuracy": majority_metrics["accuracy"],
                "macro_precision": majority_metrics["macro_precision"],
                "macro_recall": majority_metrics["macro_recall"],
                "macro_f1": majority_metrics["macro_f1"],
            },
            "TFIDF_LogisticRegression_Baseline": {
                "accuracy": tfidf_metrics["accuracy"],
                "macro_precision": tfidf_metrics["macro_precision"],
                "macro_recall": tfidf_metrics["macro_recall"],
                "macro_f1": tfidf_metrics["macro_f1"],
            },
            "BrandSupport_Semantic_Pipeline": {
                "accuracy": primary_intent_metrics["accuracy"],
                "macro_precision": primary_intent_metrics["macro_precision"],
                "macro_recall": primary_intent_metrics["macro_recall"],
                "macro_f1": primary_intent_metrics["macro_f1"],
            },
        },
        "retrieval_performance": retrieval_metrics,
        "escalation_performance": escalation_metrics,
        "reply_quality": {
            **reply_text_metrics,
            "mean_grounding_score": round(float(pd.Series(grounding_scores).mean()), 4),
        },
        "llm_judge_scores": judge_results["summary"],
        "top_curated_failure_cases": CURATED_FAILURE_CASES,
        "runtime_failure_count": len(failures),
        "primary_intent_full_metrics": primary_intent_metrics,
        "tfidf_intent_full_metrics": tfidf_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    logger.info(f"Full benchmark results successfully saved to {output_path}")
    return benchmark_report
