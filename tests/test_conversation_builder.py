"""Unit tests for conversation reconstruction and SQLite storage."""

import pytest
import sqlite3
from pathlib import Path
import pandas as pd
from src.conversation_builder import build_conversations, save_conversations_to_db, get_conversations_from_db, init_database


def test_build_conversations_chaining():
    # Construct small test DataFrame
    data = [
        {
            "tweet_id": "101",
            "author_id": "cust_1",
            "inbound": True,
            "created_at": "2026-08-20T10:00:00Z",
            "text": "@AppleSupport my battery health is 70%",
            "response_tweet_id": "102",
            "in_response_to_tweet_id": "",
        },
        {
            "tweet_id": "102",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "2026-08-20T10:05:00Z",
            "text": "@cust_1 We recommend a battery service at an Apple Store.",
            "response_tweet_id": "",
            "in_response_to_tweet_id": "101",
        }
    ]
    df = pd.DataFrame(data)
    convs = build_conversations(df, brand_name="AppleSupport")

    assert len(convs) == 1
    c = convs[0]
    assert c["conversation_id"] == "conv_101"
    assert c["brand_name"] == "AppleSupport"
    assert c["intent"] == "hardware_device_issue"
    assert "battery" in c["cleaned_customer_query"].lower()
    assert "battery service" in c["cleaned_brand_reply"].lower()


def test_sqlite_persistence(tmp_path):
    test_db = tmp_path / "test_convs.db"
    init_database(test_db)

    sample_convs = [{
        "conversation_id": "conv_999",
        "brand_name": "AppleSupport",
        "customer_author_id": "cust_999",
        "customer_query": "@AppleSupport test query",
        "cleaned_customer_query": "test query",
        "brand_reply": "@cust_999 test reply",
        "cleaned_brand_reply": "test reply",
        "turns": [],
        "num_turns": 2,
        "intent": "general_feedback_complaint",
        "created_at": "2026-08-20T12:00:00Z",
    }]

    inserted = save_conversations_to_db(sample_convs, db_path=test_db)
    assert inserted == 1

    fetched = get_conversations_from_db(brand_name="AppleSupport", db_path=test_db)
    assert len(fetched) == 1
    assert fetched[0]["conversation_id"] == "conv_999"
