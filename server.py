"""FastAPI Backend Server for BrandSupport AI Web Application."""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import (
    DEFAULT_BRAND,
    DB_PATH,
    EVAL_RESULTS_PATH,
    GOLDEN_SET_PATH,
    INTENT_LABELS,
    INTENT_DESCRIPTIONS,
    SUPPORTED_BRANDS,
)
from src.pipeline import BrandSupportPipeline
from src.conversation_builder import get_conversations_from_db
from evaluation.failure_analyzer import CURATED_FAILURE_CASES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrandSupport-Server")

app = FastAPI(
    title="BrandSupport AI API",
    description="Enterprise Trustworthy AI Customer Support Agent Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instances cache
_pipeline_cache: Dict[str, BrandSupportPipeline] = {}


def get_pipeline(brand: str = DEFAULT_BRAND) -> BrandSupportPipeline:
    if brand not in _pipeline_cache:
        try:
            _pipeline_cache[brand] = BrandSupportPipeline.load_from_artifacts(brand_name=brand)
        except Exception as e:
            logger.warning(f"Could not load saved artifacts for {brand} ({e}). Using initialized pipeline.")
            _pipeline_cache[brand] = BrandSupportPipeline(brand_name=brand)
    return _pipeline_cache[brand]


# Request models
class QueryRequest(BaseModel):
    query: str
    brand: str = DEFAULT_BRAND
    num_turns: int = 1
    intent_threshold: float = 0.65
    retrieval_threshold: float = 0.60
    grounding_threshold: float = 0.70


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/brands")
def get_brands():
    return {"supported_brands": SUPPORTED_BRANDS, "default": DEFAULT_BRAND}


@app.get("/api/intents")
def get_intents():
    return {
        "intents": INTENT_LABELS,
        "descriptions": INTENT_DESCRIPTIONS,
    }


@app.post("/api/process")
def process_message(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Customer query cannot be empty.")

    pipeline = get_pipeline(req.brand)
    pipeline.escalation_engine.intent_threshold = req.intent_threshold
    pipeline.escalation_engine.retrieval_threshold = req.retrieval_threshold
    pipeline.escalation_engine.grounding_threshold = req.grounding_threshold

    res = pipeline.process_query(
        customer_query=req.query,
        brand_name=req.brand,
        num_turns=req.num_turns,
    )
    return res


@app.get("/api/eval-results")
def get_eval_results():
    if not EVAL_RESULTS_PATH.exists():
        return {
            "available": False,
            "message": "Evaluation results not found. Run python run_pipeline.py --stage eval first."
        }
    with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["available"] = True
    return data


@app.get("/api/failure-cases")
def get_failure_cases():
    return {
        "failure_cases": CURATED_FAILURE_CASES,
        "headline_analysis": {
            "title": "What is misleading about my headline number?",
            "points": [
                {
                    "title": "Class Frequency Skew",
                    "description": "Frequent battery/complaint queries dominate datasets. High accuracy masks failures on rare catastrophic safety hazards."
                },
                {
                    "title": "Evaluation Leakage Shortcuts",
                    "description": "Random splits allow template memorization. Strict conversation-graph partitioning is mandatory for true out-of-distribution evaluation."
                },
                {
                    "title": "Asymmetric Error Costs",
                    "description": "Standard accuracy treats all errors equally. Auto-handling a battery fire hazard is drastically more severe than misclassifying a how-to question."
                },
                {
                    "title": "Multi-Turn Friction Accumulation",
                    "description": "Single-turn accuracy does not reflect multi-turn customer fatigue from repetitive bot advice."
                }
            ]
        }
    }


@app.get("/api/conversations")
def get_conversations(
    brand: Optional[str] = Query(None),
    intent: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50),
):
    if not DB_PATH.exists():
        return {"conversations": [], "count": 0}

    convs = get_conversations_from_db(
        brand_name=brand if brand and brand != "All" else None,
        intent=intent if intent and intent != "All" else None,
        limit=limit,
        db_path=DB_PATH,
    )

    if search:
        s_low = search.lower()
        convs = [c for c in convs if s_low in c.get("cleaned_customer_query", "").lower() or s_low in c.get("cleaned_brand_reply", "").lower()]

    return {"conversations": convs, "count": len(convs)}


# Mount static frontend directory
web_dir = Path(__file__).resolve().parent / "web"
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(web_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
