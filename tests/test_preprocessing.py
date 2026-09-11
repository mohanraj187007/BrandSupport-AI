"""Unit tests for preprocessing and PII masking."""

import pytest
from src.preprocessing import clean_text, extract_metadata_features


def test_clean_text_basic():
    raw = "@AppleSupport my screen is cracked &amp; broken! http://t.co/xyz123"
    cleaned = clean_text(raw)
    assert "@AppleSupport" not in cleaned
    assert "&" in cleaned
    assert "&amp;" not in cleaned
    assert "[URL]" in cleaned
    assert "broken" in cleaned


def test_pii_masking():
    raw = "@AmazonHelp email me at customer.test@gmail.com or call +1 (555) 234-5678 regarding order #112-9876543-1234567"
    cleaned = clean_text(raw, mask_pii=True)
    assert "customer.test@gmail.com" not in cleaned
    assert "[EMAIL]" in cleaned
    assert "555" not in cleaned
    assert "[PHONE]" in cleaned
    assert "[REF_NUMBER]" in cleaned or "order" in cleaned.lower()


def test_extract_metadata_features():
    text = "HELP! My battery is burning hot! URGENT!!"
    features = extract_metadata_features(text)
    assert features["is_urgent"] is True
    assert features["exclamation_count"] >= 2
    assert features["all_caps_words"] >= 2
