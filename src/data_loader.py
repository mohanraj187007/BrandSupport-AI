"""Data loader and schema validator for Twitter Customer Support dataset."""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

from src.config import DEFAULT_CSV_PATH, FULL_CSV_PATH, DEFAULT_BRAND

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def load_raw_data(file_path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Loads raw customer support dataset from CSV and validates schema.
    
    Args:
        file_path: Path to CSV file. Defaults to sample_twcs.csv or twcs.csv if present.
        
    Returns:
        Cleaned pandas DataFrame conforming to TWCS schema.
    """
    if file_path is None:
        if FULL_CSV_PATH.exists():
            file_path = FULL_CSV_PATH
        elif DEFAULT_CSV_PATH.exists():
            file_path = DEFAULT_CSV_PATH
        else:
            raise FileNotFoundError(
                f"No dataset found at {FULL_CSV_PATH} or {DEFAULT_CSV_PATH}. "
                f"Please generate the sample data first using python run_pipeline.py --stage data"
            )

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file does not exist at {path_obj}")

    logger.info(f"Loading raw dataset from {path_obj}...")
    df = pd.read_csv(
        path_obj,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": bool,
            "created_at": str,
            "text": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        low_memory=False,
    )

    # Validate schema
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV schema mismatch. Missing required columns: {missing}")

    # Standardize types and handle missing values
    df["tweet_id"] = df["tweet_id"].astype(str).str.strip()
    df["author_id"] = df["author_id"].astype(str).str.strip()
    df["inbound"] = df["inbound"].fillna(False).astype(bool)
    df["text"] = df["text"].fillna("").astype(str)
    df["response_tweet_id"] = df["response_tweet_id"].fillna("").astype(str)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str)

    logger.info(f"Loaded {len(df):,} total tweets from dataset.")
    return df


def get_available_brands(df: pd.DataFrame) -> Dict[str, int]:
    """
    Identifies all brand handles present in the dataset and their activity count.
    Brand handles are author_ids for outbound tweets (inbound == False).
    """
    outbound = df[~df["inbound"]]
    brand_counts = outbound["author_id"].value_counts().to_dict()
    return brand_counts


def filter_by_brand(df: pd.DataFrame, brand_name: str = DEFAULT_BRAND) -> pd.DataFrame:
    """
    Filters conversation tweets associated with a specific brand.
    Includes both inbound customer tweets mentioning/targeting the brand and outbound replies by the brand.
    """
    brand_clean = brand_name.lstrip("@")
    
    # 1. Outbound tweets by the brand
    brand_outbound = df[(~df["inbound"]) & (df["author_id"].str.lower() == brand_clean.lower())]
    
    # 2. Inbound tweets mentioning the brand or responded to by the brand
    inbound_mentions = df[
        df["inbound"] & (df["text"].str.contains(f"@{brand_clean}", case=False, regex=False))
    ]
    
    # 3. Inbound tweets that are referenced in in_response_to_tweet_id or response_tweet_id
    referenced_ids = set()
    for resp_ids in brand_outbound["response_tweet_id"]:
        if resp_ids and resp_ids != "nan":
            for tid in resp_ids.split(","):
                referenced_ids.add(tid.strip())
                
    for in_resp in brand_outbound["in_response_to_tweet_id"]:
        if in_resp and in_resp != "nan":
            referenced_ids.add(in_resp.strip())

    referenced_tweets = df[df["tweet_id"].isin(referenced_ids)]

    brand_df = pd.concat([brand_outbound, inbound_mentions, referenced_tweets]).drop_duplicates(subset=["tweet_id"])
    logger.info(f"Filtered {len(brand_df):,} tweets for brand @{brand_clean}")
    return brand_df.sort_values(by="tweet_id")
