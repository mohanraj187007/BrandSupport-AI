"""Multi-turn conversation builder and SQLite database repository."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config import DB_PATH, DEFAULT_BRAND, INTENT_LABELS
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


def init_database(db_path: Path = DB_PATH) -> None:
    """Initializes the SQLite database schema for storing reconstructed conversations."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE,
                brand_name TEXT,
                customer_author_id TEXT,
                customer_query TEXT,
                cleaned_customer_query TEXT,
                brand_reply TEXT,
                cleaned_brand_reply TEXT,
                turns_json TEXT,
                num_turns INTEGER,
                intent TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_brand ON conversations(brand_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intent ON conversations(intent);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_id ON conversations(conversation_id);")
        conn.commit()
    logger.info(f"Initialized conversations database at {db_path}")


def infer_rule_based_intent(text: str) -> str:
    """
    Robust rule-based heuristic for intent tagging during dataset ingestion.
    Matches customer domain signals to one of the 8 core intents.
    """
    t = text.lower()

    if any(k in t for k in ["battery", "screen", "display", "cracked", "charger", "charging", "overheating", "speaker", "magsafe", "hardware", "trackpad", "won't turn on"]):
        return "hardware_device_issue"

    if any(k in t for k in ["update", "ios", "ipados", "macos", "crash", "freeze", "freezing", "bug", "glitch", "boot loop", "recovery mode", "stuck on apple logo"]):
        return "software_update_bug"

    if any(k in t for k in ["apple id", "password", "locked", "billing", "charge", "charged", "subscription", "payment", "receipt", "account recovery", "2fa", "two-factor"]):
        return "account_auth_billing"

    if any(k in t for k in ["order #", "shipping", "shipped", "tracking", "courier", "package", "delivered", "delivery", "orderstatus"]):
        return "order_shipping_delivery"

    if any(k in t for k in ["refund", "return", "exchange", "replacement", "trade-in refund", "holiday return"]):
        return "returns_refund_exchange"

    if any(k in t for k in ["wifi", "wi-fi", "bluetooth", "airpods", "cellular", "no service", "signal", "pairing", "disconnect", "sync", "icloud photo"]):
        return "connectivity_sync_bluetooth"

    if any(k in t for k in ["how to", "how do i", "transfer", "backup", "family sharing", "standby mode", "personal hotspot", "settings", "configure"]):
        return "feature_how_to_inquiry"

    if any(k in t for k in ["feedback", "complaint", "appointment", "genius bar", "store staff", "compliment", "service"]):
        return "general_feedback_complaint"

    return "general_feedback_complaint"


def build_conversations(df: pd.DataFrame, brand_name: str = DEFAULT_BRAND) -> List[Dict[str, Any]]:
    """
    Reconstructs structured multi-turn customer support conversations from raw tweet links.
    
    Args:
        df: DataFrame containing Twitter Customer Support records.
        brand_name: Brand handle to filter for.
        
    Returns:
        List of structured conversation dictionaries.
    """
    brand_clean = brand_name.lstrip("@").lower()
    
    # Map tweet_id to row for O(1) lookup
    tweet_dict = {str(row["tweet_id"]): row for _, row in df.iterrows()}
    
    conversations = []
    processed_roots = set()

    # Identify brand replies (outbound tweets from this brand)
    for _, row in df.iterrows():
        is_brand_tweet = (not row["inbound"]) and (str(row["author_id"]).lower() == brand_clean)
        if not is_brand_tweet:
            continue

        brand_tweet_id = str(row["tweet_id"])
        in_reply_to_id = str(row.get("in_response_to_tweet_id", "")).strip()

        if not in_reply_to_id or in_reply_to_id == "nan" or in_reply_to_id not in tweet_dict:
            continue

        customer_tweet = tweet_dict[in_reply_to_id]
        if not customer_tweet["inbound"]:
            continue

        # Trace root tweet if in_reply_to_id is part of a longer chain
        root_tweet = customer_tweet
        visited = {customer_tweet["tweet_id"]}
        while str(root_tweet.get("in_response_to_tweet_id", "")).strip() in tweet_dict:
            parent_id = str(root_tweet["in_response_to_tweet_id"]).strip()
            if parent_id in visited:
                break
            visited.add(parent_id)
            parent_tweet = tweet_dict[parent_id]
            if parent_tweet["inbound"]:
                root_tweet = parent_tweet
            else:
                break

        root_id = str(root_tweet["tweet_id"])
        if root_id in processed_roots:
            continue
        processed_roots.add(root_id)

        # Build dialog turns
        turns = [
            {
                "speaker": "customer",
                "author_id": str(root_tweet["author_id"]),
                "text": str(root_tweet["text"]),
                "created_at": str(root_tweet.get("created_at", "")),
            },
            {
                "speaker": "brand",
                "author_id": brand_name,
                "text": str(row["text"]),
                "created_at": str(row.get("created_at", "")),
            }
        ]

        # Check for subsequent customer replies
        subsequent_ids = str(row.get("response_tweet_id", "")).split(",")
        for sub_id in subsequent_ids:
            sub_id = sub_id.strip()
            if sub_id in tweet_dict:
                sub_tweet = tweet_dict[sub_id]
                turns.append({
                    "speaker": "customer" if sub_tweet["inbound"] else "brand",
                    "author_id": str(sub_tweet["author_id"]),
                    "text": str(sub_tweet["text"]),
                    "created_at": str(sub_tweet.get("created_at", "")),
                })

        raw_cust = str(root_tweet["text"])
        raw_brand = str(row["text"])
        clean_cust = clean_text(raw_cust)
        clean_brand = clean_text(raw_brand)

        if not clean_cust or not clean_brand:
            continue

        intent = infer_rule_based_intent(clean_cust)

        conversations.append({
            "conversation_id": f"conv_{root_id}",
            "brand_name": brand_name,
            "customer_author_id": str(root_tweet["author_id"]),
            "customer_query": raw_cust,
            "cleaned_customer_query": clean_cust,
            "brand_reply": raw_brand,
            "cleaned_brand_reply": clean_brand,
            "turns": turns,
            "num_turns": len(turns),
            "intent": intent,
            "created_at": str(root_tweet.get("created_at", "")),
        })

    logger.info(f"Reconstructed {len(conversations)} multi-turn conversations for {brand_name}.")
    return conversations


def save_conversations_to_db(conversations: List[Dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Inserts or updates conversations into the SQLite database."""
    init_database(db_path)
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for conv in conversations:
            cursor.execute("""
                INSERT OR REPLACE INTO conversations (
                    conversation_id, brand_name, customer_author_id,
                    customer_query, cleaned_customer_query,
                    brand_reply, cleaned_brand_reply,
                    turns_json, num_turns, intent, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conv["conversation_id"],
                conv["brand_name"],
                conv["customer_author_id"],
                conv["customer_query"],
                conv["cleaned_customer_query"],
                conv["brand_reply"],
                conv["cleaned_brand_reply"],
                json.dumps(conv.get("turns", [])),
                conv["num_turns"],
                conv["intent"],
                conv["created_at"],
            ))
            inserted += 1
        conn.commit()
    logger.info(f"Persisted {inserted} conversations to {db_path}.")
    return inserted


def get_conversations_from_db(
    brand_name: Optional[str] = None,
    intent: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Retrieves conversations from SQLite database with optional filters."""
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []

        if brand_name:
            query += " AND LOWER(brand_name) = ?"
            params.append(brand_name.lower())
        if intent:
            query += " AND intent = ?"
            params.append(intent)

        query += " ORDER BY id ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            d = dict(row)
            d["turns"] = json.loads(d["turns_json"]) if d.get("turns_json") else []
            results.append(d)
        return results
