"""LLM-as-a-Judge Evaluation Module for Customer Support Agent outputs."""

import json
import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Evaluates response quality across 5 standard customer support dimensions:
    1. Relevance (1-5)
    2. Grounding / Faithfulness (1-5)
    3. Helpfulness (1-5)
    4. Unsupported Claims / Hallucinations (0 or 1)
    5. Escalation Correctness (1-5)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_MODEL):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client for LLMJudge with model {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for judge ({e}). Using deterministic rubric.")
                self.client = None

    def _evaluate_deterministic(
        self,
        customer_query: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str,
        decision: str,
        ground_truth_escalation: str,
        grounding_score: float,
    ) -> Dict[str, Any]:
        """
        Deterministic rubric-based scoring fallback that ensures reproducible, zero-cost evaluation.
        """
        query_clean = clean_text(customer_query).lower()
        reply_clean = clean_text(generated_reply).lower()

        # 1. Relevance Score (1-5)
        # Check topic overlap between customer query and reply
        q_words = set(re.findall(r"\b[a-z]{3,}\b", query_clean))
        r_words = set(re.findall(r"\b[a-z]{3,}\b", reply_clean))
        overlap_ratio = len(q_words.intersection(r_words)) / max(1, len(q_words))
        
        if overlap_ratio > 0.4:
            relevance = 5.0
        elif overlap_ratio > 0.25:
            relevance = 4.0
        elif overlap_ratio > 0.1:
            relevance = 3.0
        else:
            relevance = 2.0

        # 2. Grounding Score (1-5)
        if grounding_score >= 0.75:
            faithfulness = 5.0
        elif grounding_score >= 0.50:
            faithfulness = 4.0
        elif grounding_score >= 0.30:
            faithfulness = 3.0
        else:
            faithfulness = 2.0

        # 3. Helpfulness (1-5)
        has_actionable_url = any(u in reply_clean for u in ["apple.com", "amazon", "settings", "reportaproblem", "iforgot", "reset", "mode", "appointment"])
        has_step_words = any(w in reply_clean for w in ["press", "select", "tap", "go to", "restart", "connect", "check", "visit"])
        
        if has_actionable_url and has_step_words:
            helpfulness = 5.0
        elif has_actionable_url or has_step_words:
            helpfulness = 4.0
        elif len(reply_clean) > 20:
            helpfulness = 3.5
        else:
            helpfulness = 2.0

        # 4. Unsupported Claims / Hallucinations (0 or 1)
        # Check for fabricated promises (e.g., instant free money, guaranteed 100% refund)
        unsupported = 0
        forbidden_promises = ["100% refund", "free replacement guaranteed", "instant cash back", "waiving all fees", "guaranteed delivery in 1 hour"]
        for p in forbidden_promises:
            if p in reply_clean:
                unsupported = 1
                break

        # 5. Escalation Correctness (1-5)
        if decision == ground_truth_escalation:
            escalation_score = 5.0
        else:
            # Penalize missed escalation more severely than unnecessary escalation
            if ground_truth_escalation == "ESCALATE_TO_HUMAN" and decision == "AUTO_HANDLE":
                escalation_score = 1.0  # Dangerous miss
            else:
                escalation_score = 3.0  # False alarm / safe conservative routing

        return {
            "relevance": relevance,
            "faithfulness": faithfulness,
            "helpfulness": helpfulness,
            "unsupported_claims": unsupported,
            "escalation_correctness": escalation_score,
            "judge_method": "deterministic_rubric",
        }

    def evaluate_sample(
        self,
        customer_query: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str,
        decision: str,
        ground_truth_escalation: str,
        grounding_score: float,
    ) -> Dict[str, Any]:
        """Evaluates a single sample using LLM or deterministic fallback."""
        if self.client is None:
            return self._evaluate_deterministic(
                customer_query, retrieved_evidence, generated_reply,
                decision, ground_truth_escalation, grounding_score
            )

        evidence_text = "\n".join([f"- {e.get('cleaned_brand_reply')}" for e in retrieved_evidence])
        judge_prompt = f"""
You are an expert AI Customer Support Quality Auditor.
Evaluate the following customer support interaction strictly based on the rubric.

[Customer Query]
{customer_query}

[Retrieved Support Evidence]
{evidence_text}

[Generated Agent Reply]
{generated_reply}

[System Escalation Decision]
{decision}

[Ground Truth Expected Escalation]
{ground_truth_escalation}

Rubric:
1. Relevance (1 to 5): Does the reply directly address the customer's query?
2. Faithfulness / Grounding (1 to 5): Is the reply fully supported by the retrieved evidence without inventing unsupported facts?
3. Helpfulness (1 to 5): Does it offer clear, practical guidance or links?
4. Unsupported Claims (0 or 1): 1 if the agent fabricated unauthorized refund guarantees or false promises; 0 if safe.
5. Escalation Correctness (1 to 5): Did the agent choose the right decision (AUTO_HANDLE vs ESCALATE_TO_HUMAN)?

Return ONLY a valid JSON object matching:
{{
  "relevance": <1.0-5.0>,
  "faithfulness": <1.0-5.0>,
  "helpfulness": <1.0-5.0>,
  "unsupported_claims": <0 or 1>,
  "escalation_correctness": <1.0-5.0>
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            res_json = json.loads(response.choices[0].message.content)
            res_json["judge_method"] = "llm_openai"
            return res_json
        except Exception as e:
            logger.warning(f"OpenAI judge call failed ({e}), using deterministic fallback.")
            return self._evaluate_deterministic(
                customer_query, retrieved_evidence, generated_reply,
                decision, ground_truth_escalation, grounding_score
            )

    def evaluate_batch(self, evaluation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates a batch of test records and aggregates scores."""
        scores = {
            "relevance": [],
            "faithfulness": [],
            "helpfulness": [],
            "unsupported_claims": [],
            "escalation_correctness": [],
        }

        detailed_results = []
        for rec in evaluation_records:
            judge_res = self.evaluate_sample(
                customer_query=rec["customer_query"],
                retrieved_evidence=rec.get("retrieved_evidence", []),
                generated_reply=rec.get("generated_reply", ""),
                decision=rec.get("decision", "AUTO_HANDLE"),
                ground_truth_escalation=rec.get("ground_truth_escalation", "AUTO_HANDLE"),
                grounding_score=rec.get("grounding_score", 1.0),
            )
            detailed_results.append({**rec, **judge_res})
            for k in scores.keys():
                scores[k].append(float(judge_res.get(k, 0.0)))

        aggregated = {
            "mean_relevance": round(float(np.mean(scores["relevance"])), 2),
            "mean_faithfulness": round(float(np.mean(scores["faithfulness"])), 2),
            "mean_helpfulness": round(float(np.mean(scores["helpfulness"])), 2),
            "unsupported_claims_rate": round(float(np.mean(scores["unsupported_claims"])), 4),
            "mean_escalation_correctness": round(float(np.mean(scores["escalation_correctness"])), 2),
            "total_samples_judged": len(evaluation_records),
        }
        return {
            "summary": aggregated,
            "details": detailed_results,
        }
