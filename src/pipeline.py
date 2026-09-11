"""Unified End-to-End Inference Pipeline for BrandSupport AI."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import (
    DEFAULT_BRAND,
    SEMANTIC_CLASSIFIER_PATH,
    FAISS_INDEX_PATH,
    TFIDF_MODEL_PATH,
    INTENT_DESCRIPTIONS,
)
from src.preprocessing import clean_text, extract_metadata_features
from src.intent_classifier import SemanticIntentClassifier
from src.retriever import FAISSRetriever
from src.reply_generator import GroundedReplyGenerator
from src.escalation import EscalationEngine

logger = logging.getLogger(__name__)


class BrandSupportPipeline:
    """
    Unified end-to-end inference pipeline orchestrating intent classification,
    FAISS historical retrieval, grounded answer generation, and escalation routing.
    """

    def __init__(
        self,
        brand_name: str = DEFAULT_BRAND,
        intent_classifier: Optional[SemanticIntentClassifier] = None,
        retriever: Optional[FAISSRetriever] = None,
        reply_generator: Optional[GroundedReplyGenerator] = None,
        escalation_engine: Optional[EscalationEngine] = None,
    ):
        self.brand_name = brand_name
        self.intent_classifier = intent_classifier or SemanticIntentClassifier()
        self.retriever = retriever or FAISSRetriever()
        
        # Share encoder across classifier and retriever to save memory and avoid redundant downloads
        if self.intent_classifier.encoder is not None and self.retriever.encoder is None:
            self.retriever.encoder = self.intent_classifier.encoder
        elif self.retriever.encoder is not None and self.intent_classifier.encoder is None:
            self.intent_classifier.encoder = self.retriever.encoder

        self.reply_generator = reply_generator or GroundedReplyGenerator()
        self.escalation_engine = escalation_engine or EscalationEngine()

    @classmethod
    def load_from_artifacts(
        cls,
        brand_name: str = DEFAULT_BRAND,
        classifier_path: Path = SEMANTIC_CLASSIFIER_PATH,
        index_path: Path = FAISS_INDEX_PATH,
    ) -> "BrandSupportPipeline":
        """Loads trained model artifacts and indices from disk with shared encoder."""
        logger.info(f"Loading pipeline artifacts for brand {brand_name}...")
        classifier = SemanticIntentClassifier.load(classifier_path)
        retriever = FAISSRetriever.load(index_path)
        
        # Share encoder instance
        if classifier.encoder is not None:
            retriever.encoder = classifier.encoder

        reply_gen = GroundedReplyGenerator()
        escalation = EscalationEngine()

        return cls(
            brand_name=brand_name,
            intent_classifier=classifier,
            retriever=retriever,
            reply_generator=reply_gen,
            escalation_engine=escalation,
        )

    def process_query(
        self,
        customer_query: str,
        brand_name: Optional[str] = None,
        num_turns: int = 1,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes complete end-to-end inference flow for a customer query.
        
        Args:
            customer_query: Raw input customer message.
            brand_name: Target brand override.
            num_turns: Number of interaction turns in conversation.
            top_k: Number of historical cases to retrieve.
            
        Returns:
            Comprehensive structured prediction and decision dictionary.
        """
        target_brand = brand_name or self.brand_name

        # 1. Preprocessing & Text Cleaning
        cleaned_query = clean_text(customer_query)
        meta_features = extract_metadata_features(customer_query)

        # 2. Intent Classification & Confidence Calibration
        intent_result = self.intent_classifier.predict_single(customer_query)
        predicted_intent = intent_result["predicted_intent"]
        confidence = intent_result["confidence"]

        # 3. FAISS Historical Retrieval
        retrieved_evidence = self.retriever.retrieve(
            query=customer_query,
            top_k=top_k,
            brand_name=target_brand,
        )

        # 4. Evidence-Grounded Reply Generation
        reply_result = self.reply_generator.generate(
            customer_query=customer_query,
            predicted_intent=predicted_intent,
            evidence_list=retrieved_evidence,
            brand_name=target_brand,
        )
        generated_reply = reply_result["reply"]
        grounding_score = reply_result["grounding_score"]

        # 5. Escalation Decision Engine
        escalation_result = self.escalation_engine.evaluate(
            customer_query=customer_query,
            predicted_intent=predicted_intent,
            intent_confidence=confidence,
            evidence_list=retrieved_evidence,
            grounding_score=grounding_score,
            num_turns=num_turns,
        )

        return {
            "brand": target_brand,
            "raw_query": customer_query,
            "cleaned_query": cleaned_query,
            "predicted_intent": predicted_intent,
            "intent_description": INTENT_DESCRIPTIONS.get(predicted_intent, ""),
            "intent_confidence": confidence,
            "intent_margin": intent_result.get("margin", 0.0),
            "all_intent_probabilities": intent_result.get("all_probabilities", {}),
            "retrieved_evidence": retrieved_evidence,
            "generated_reply": generated_reply,
            "grounding_score": grounding_score,
            "generation_mode": reply_result.get("generation_mode", "local_extractive"),
            "decision": escalation_result["decision"],
            "is_escalated": escalation_result["is_escalated"],
            "escalation_reasons": escalation_result["reasons"],
            "recommended_queue": escalation_result["recommended_queue"],
            "risk_level": escalation_result["risk_level"],
            "aggregate_confidence": escalation_result["aggregate_confidence"],
            "metadata_features": meta_features,
        }
