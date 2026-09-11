"""Evidence-grounded reply generator with LLM and deterministic local synthesis."""

import logging
import os
import re
from typing import List, Dict, Any, Optional
from src.config import OPENAI_API_KEY, OPENAI_MODEL, LLM_TEMPERATURE, GROUNDING_THRESHOLD
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


class GroundedReplyGenerator:
    """
    Synthesizes customer support replies strictly grounded in retrieved historical resolutions.
    Enforces anti-hallucination guardrails forbidding fabricated policies, refunds, or false promises.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_MODEL):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client for GroundedReplyGenerator with model {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client ({e}). Local generator will be used.")
                self.client = None

    def calculate_grounding_score(self, reply: str, evidence_list: List[Dict[str, Any]]) -> float:
        """
        Calculates lexical and factual grounding score between the draft reply and retrieved evidence.
        Score ranges from 0.0 (ungrounded / hallucinated) to 1.0 (fully grounded).
        """
        if not evidence_list:
            return 0.0

        reply_clean = clean_text(reply).lower()
        reply_tokens = set(re.findall(r"\b[a-z]{3,}\b", reply_clean))
        if not reply_tokens:
            return 0.0

        # Aggregate evidence tokens from retrieved brand resolutions
        evidence_text = " ".join([e.get("cleaned_brand_reply", "") for e in evidence_list]).lower()
        evidence_tokens = set(re.findall(r"\b[a-z]{3,}\b", evidence_text))

        if not evidence_tokens:
            return 0.0

        # Calculate Jaccard / token overlap
        overlap = reply_tokens.intersection(evidence_tokens)
        coverage = len(overlap) / len(reply_tokens)
        
        # Penalize if unsupported promise keywords appear in reply but not in evidence
        unsupported_terms = ["100% refund", "free replacement guaranteed", "instant cash back", "waiving all fees", "guaranteed delivery in 1 hour"]
        hallucination_penalty = 0.0
        for term in unsupported_terms:
            if term in reply_clean and term not in evidence_text:
                hallucination_penalty += 0.35

        score = max(0.0, min(1.0, coverage - hallucination_penalty))
        return round(score, 4)

    def _generate_deterministic_reply(
        self,
        customer_query: str,
        predicted_intent: str,
        evidence_list: List[Dict[str, Any]],
        brand_name: str,
    ) -> str:
        """
        Generates a reliable, strictly grounded reply locally without requiring external API calls.
        Extracts verified steps, official support links, and guidance directly from historical resolutions.
        """
        if not evidence_list:
            return f"Thank you for contacting @{brand_name}. We would like to look into this for you. Please share your account details or direct message us so our team can assist."

        top_evidence = evidence_list[0]
        top_reply = top_evidence.get("cleaned_brand_reply", "")

        # Extract URLs / official portals if present
        urls = re.findall(r"\[URL\]|https?://\S+|reportaproblem\.apple\.com|iforgot\.apple\.com|apple\.com/\S+", top_evidence.get("brand_reply", ""))
        url_mention = f" ({urls[0]})" if urls and "[URL]" not in urls[0] else ""

        # Format clean, helpful response based on verified historical response
        cleaned_body = re.sub(r"^@\w+\s*", "", top_reply).strip()
        
        # Ensure brand signoff / opening
        reply = f"{cleaned_body}{url_mention}"
        return reply

    def _generate_llm_reply(
        self,
        customer_query: str,
        predicted_intent: str,
        evidence_list: List[Dict[str, Any]],
        brand_name: str,
    ) -> str:
        """Calls LLM API with strict grounding system prompt."""
        evidence_snippets = "\n\n".join([
            f"[Historical Case #{i+1} - Similarity: {e.get('similarity_score', 0):.2f}]\n"
            f"Customer: {e.get('cleaned_customer_query')}\n"
            f"Brand Resolution: {e.get('cleaned_brand_reply')}"
            for i, e in enumerate(evidence_list)
        ])

        system_prompt = (
            f"You are an official customer support agent for @{brand_name}.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. You must answer the customer's query strictly using ONLY the factual guidance provided in the Historical Resolutions.\n"
            "2. NEVER invent policies, refund promises, delivery guarantees, or timelines not present in the evidence.\n"
            "3. If the evidence does not provide an exact solution, direct the customer to the official support portal or advise them to DM support.\n"
            "4. Keep the tone concise, empathetic, and professional (under 280 characters if possible)."
        )

        user_prompt = (
            f"Brand: {brand_name}\n"
            f"Customer Intent: {predicted_intent}\n"
            f"Customer Message: {customer_query}\n\n"
            f"Retrieved Historical Evidence:\n{evidence_snippets}\n\n"
            "Grounded Customer Reply:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    def generate(
        self,
        customer_query: str,
        predicted_intent: str,
        evidence_list: List[Dict[str, Any]],
        brand_name: str = "AppleSupport",
    ) -> Dict[str, Any]:
        """
        Generates reply and calculates grounding faithfulness score.
        
        Returns:
            Dict with 'reply', 'grounding_score', 'used_evidence', 'generation_mode'.
        """
        generation_mode = "local_extractive"
        reply = ""

        if self.client is not None and evidence_list:
            try:
                reply = self._generate_llm_reply(customer_query, predicted_intent, evidence_list, brand_name)
                generation_mode = "llm_grounded"
            except Exception as e:
                logger.warning(f"LLM generation failed ({e}), falling back to deterministic synthesis.")
                reply = self._generate_deterministic_reply(customer_query, predicted_intent, evidence_list, brand_name)
        else:
            reply = self._generate_deterministic_reply(customer_query, predicted_intent, evidence_list, brand_name)

        grounding_score = self.calculate_grounding_score(reply, evidence_list)

        return {
            "reply": reply,
            "grounding_score": grounding_score,
            "used_evidence": evidence_list[:2] if evidence_list else [],
            "generation_mode": generation_mode,
        }
