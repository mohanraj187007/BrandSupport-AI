"""Configuration and constants for BrandSupport AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
DEFAULT_CSV_PATH = DATA_DIR / "sample_twcs.csv"
FULL_CSV_PATH = DATA_DIR / "twcs.csv"
DB_PATH = DATA_DIR / "conversations.db"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.joblib"
SEMANTIC_CLASSIFIER_PATH = MODELS_DIR / "semantic_classifier.joblib"
GOLDEN_SET_PATH = DATA_DIR / "golden_set.json"
EVAL_RESULTS_PATH = EVAL_DIR / "benchmark_results.json"

# Target default brand
DEFAULT_BRAND = os.getenv("BRAND_NAME", "AppleSupport")

# Supported Brands
SUPPORTED_BRANDS = [
    "AppleSupport",
    "AmazonHelp",
    "SpotifyCares",
    "Uber_Support",
    "Delta",
    "AmericanAir",
    "Tesco",
    "AskTarget",
]

# 8 Core Customer Support Intents
INTENT_LABELS = [
    "hardware_device_issue",
    "software_update_bug",
    "account_auth_billing",
    "order_shipping_delivery",
    "returns_refund_exchange",
    "connectivity_sync_bluetooth",
    "feature_how_to_inquiry",
    "general_feedback_complaint",
]

INTENT_DESCRIPTIONS = {
    "hardware_device_issue": "Physical device failures, battery degradation, screen issues, charging port faults, overheating.",
    "software_update_bug": "Operating system crashes, update installation failures, app glitches, boot loops, system freeze.",
    "account_auth_billing": "Apple ID / Account locked, unauthorized subscription charges, password resets, payment verification.",
    "order_shipping_delivery": "Delivery status, delayed shipments, courier tracking, incorrect address, missing parcels.",
    "returns_refund_exchange": "Return requests, replacement eligibility, refund processing timelines, store credit.",
    "connectivity_sync_bluetooth": "Wi-Fi disconnections, Bluetooth pairing errors, iCloud sync issues, network timeouts.",
    "feature_how_to_inquiry": "How-to instructions, settings navigation, iCloud backup restore, transfer data to new device.",
    "general_feedback_complaint": "Dissatisfaction with customer service, long wait times, store experience, compliments.",
}

# Escalation Trigger Keywords (Strict safety / legal / fraud / urgency routing)
HIGH_URGENCY_KEYWORDS = [
    "lawyer", "attorney", "legal action", "sue", "court", "lawsuit",
    "fraud", "scam", "unauthorized charge", "stolen card", "hacked", "breach",
    "safety hazard", "battery exploded", "smoke", "burned", "fire", "electric shock",
    "police", "emergency", "urgent help", "unacceptable service", "better business bureau", "bbb complaint"
]

# Escalation Routing Queues
ESCALATION_QUEUES = {
    "hardware_device_issue": "Tier-2 Hardware & Genius Bar Specialists",
    "software_update_bug": "Tier-2 iOS / Software Engineering Support",
    "account_auth_billing": "Security, Fraud & Billing Escalation Team",
    "order_shipping_delivery": "Logistics & Priority Fulfillment Team",
    "returns_refund_exchange": "Customer Retention & Refunds Management",
    "connectivity_sync_bluetooth": "Tier-2 Network & Peripherals Support",
    "feature_how_to_inquiry": "General Support Specialist",
    "general_feedback_complaint": "Executive Customer Relations / Senior Supervisor",
}

# Confidence & Quality Thresholds
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.65"))
RETRIEVAL_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.60"))
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.70"))

# Model settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 3

# LLM API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
