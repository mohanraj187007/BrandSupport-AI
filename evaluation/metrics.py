"""Evaluation metrics computation for Intent, Retrieval, Escalation, and Reply Quality."""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_intent_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """Computes multiclass intent classification performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": cm,
        "labels": labels,
        "per_class_report": report,
    }


def compute_retrieval_metrics(
    retrieved_results: List[List[Dict[str, Any]]],
    ground_truth_intents: List[str],
    top_k_list: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Computes Retrieval Recall@K and MRR based on whether retrieved items match ground truth intent.
    """
    recalls = {f"recall@{k}": 0.0 for k in top_k_list}
    mrr_total = 0.0
    n = len(retrieved_results)

    if n == 0:
        return {**recalls, "mrr": 0.0}

    for res_list, target_intent in zip(retrieved_results, ground_truth_intents):
        ranks = []
        for rank_idx, doc in enumerate(res_list):
            if doc.get("intent") == target_intent:
                ranks.append(rank_idx + 1)

        # Recall@K
        for k in top_k_list:
            top_k_intents = [doc.get("intent") for doc in res_list[:k]]
            if target_intent in top_k_intents:
                recalls[f"recall@{k}"] += 1.0

        # MRR
        if ranks:
            mrr_total += 1.0 / ranks[0]

    metrics = {k: round(v / n, 4) for k, v in recalls.items()}
    metrics["mrr"] = round(mrr_total / n, 4)
    return metrics


def compute_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes binary escalation routing metrics (AUTO_HANDLE vs ESCALATE_TO_HUMAN).
    Positive class = ESCALATE_TO_HUMAN.
    """
    # Map to binary (1 for ESCALATE_TO_HUMAN, 0 for AUTO_HANDLE)
    bin_true = [1 if y == "ESCALATE_TO_HUMAN" else 0 for y in y_true]
    bin_pred = [1 if y == "ESCALATE_TO_HUMAN" else 0 for y in y_pred]

    acc = accuracy_score(bin_true, bin_pred)
    prec = precision_score(bin_true, bin_pred, zero_division=0)
    rec = recall_score(bin_true, bin_pred, zero_division=0)
    f1 = f1_score(bin_true, bin_pred, zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # TN: Correctly Auto-Handled
    # FP: Unnecessary Escalation (Agent routed when it could handle)
    # FN: Missed Escalation (DANGEROUS: Bot tried to handle when human needed)
    # TP: Correctly Escalated
    cm = confusion_matrix(bin_true, bin_pred, labels=[0, 1]).tolist()
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    # Cost-weighted safety penalty (Missing escalation is 5x more costly than false escalation)
    safety_cost = (fn * 5) + (fp * 1)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "true_negatives_auto_handled": tn,
        "false_positives_unnecessary_escalation": fp,
        "false_negatives_missed_escalation": fn,
        "true_positives_correctly_escalated": tp,
        "safety_risk_cost": safety_cost,
        "confusion_matrix": cm,
    }


def compute_reply_text_metrics(generated_replies: List[str], target_replies: List[str]) -> Dict[str, float]:
    """Computes lexical token overlap and Jaccard similarity metrics against target replies."""
    scores = []
    for gen, tgt in zip(generated_replies, target_replies):
        gen_tokens = set(gen.lower().split())
        tgt_tokens = set(tgt.lower().split())
        if not gen_tokens or not tgt_tokens:
            scores.append(0.0)
            continue
        jaccard = len(gen_tokens.intersection(tgt_tokens)) / len(gen_tokens.union(tgt_tokens))
        scores.append(jaccard)

    return {
        "mean_jaccard_overlap": round(float(np.mean(scores)), 4),
        "min_overlap": round(float(np.min(scores)), 4),
        "max_overlap": round(float(np.max(scores)), 4),
    }
