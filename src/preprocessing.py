"""Text cleaning, preprocessing, and PII masking utilities."""

import re
import html
import unicodedata
from typing import Dict, Any


def clean_text(text: str, mask_pii: bool = True, strip_handles: bool = True) -> str:
    """
    Cleans raw customer or brand text from Twitter customer support.
    
    Args:
        text: Raw text string.
        mask_pii: Whether to mask emails, phone numbers, and order numbers.
        strip_handles: Whether to remove leading and inline @mentions.
        
    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unescape HTML entities (e.g. &amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # Normalize unicode characters
    cleaned = unicodedata.normalize("NFKD", cleaned)

    # Remove URLs (t.co and standard URLs)
    cleaned = re.sub(r"https?://\S+|www\.\S+", "[URL]", cleaned)

    if mask_pii:
        # Mask emails
        cleaned = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", cleaned)
        # Mask phone numbers (US/International standard patterns)
        cleaned = re.sub(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE]", cleaned)
        # Mask credit cards / 16 digit numbers
        cleaned = re.sub(r"\b(?:\d{4}[ -]?){3}\d{4}\b", "[CARD_NUMBER]", cleaned)
        # Mask order / serial / case reference numbers (e.g. #112-9876543-1234567, Case: 10023456)
        cleaned = re.sub(r"(?:order|case|ticket|ref|tracking)[\s#:]*([A-Za-z0-9\-]{6,25})", "[REF_NUMBER]", cleaned, flags=re.IGNORECASE)

    if strip_handles:
        # Strip Twitter @handles
        cleaned = re.sub(r"@\w+", "", cleaned)

    # Normalize whitespace & remove special control characters
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_metadata_features(text: str) -> Dict[str, Any]:
    """
    Extracts high-level stylistic and urgency features from the customer text.
    """
    text_len = len(text)
    word_count = len(text.split())
    has_question = "?" in text
    exclamation_count = text.count("!")
    all_caps_words = len([w for w in text.split() if w.isupper() and len(w) > 1])
    is_urgent = bool(exclamation_count >= 2 or all_caps_words >= 2 or "urgent" in text.lower())

    return {
        "text_length": text_len,
        "word_count": word_count,
        "has_question": has_question,
        "exclamation_count": exclamation_count,
        "all_caps_words": all_caps_words,
        "is_urgent": is_urgent,
    }
