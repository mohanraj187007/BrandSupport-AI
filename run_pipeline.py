"""Command-line interface (CLI) runner for BrandSupport AI."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DEFAULT_BRAND,
    DEFAULT_CSV_PATH,
    FULL_CSV_PATH,
    DB_PATH,
    SEMANTIC_CLASSIFIER_PATH,
    TFIDF_MODEL_PATH,
    FAISS_INDEX_PATH,
    GOLDEN_SET_PATH,
    EVAL_RESULTS_PATH,
    SUPPORTED_BRANDS,
)
from src.dataset_generator import generate_sample_dataset, generate_golden_evaluation_set
from src.data_loader import load_raw_data, get_available_brands, filter_by_brand
from src.conversation_builder import build_conversations, save_conversations_to_db, get_conversations_from_db
from baselines.tfidf_baseline import TFIDFIntentClassifier
from src.intent_classifier import SemanticIntentClassifier
from src.retriever import FAISSRetriever
from evaluation.evaluator import run_full_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BrandSupport-CLI")


def stage_data(brand_name: str = DEFAULT_BRAND, custom_data_path: Path = None):
    """Executes data ingestion, multi-turn reconstruction, and SQLite persistence."""
    logger.info("=" * 60)
    logger.info(f"STAGE 1: DATA PREPARATION FOR BRAND @{brand_name}")
    logger.info("=" * 60)

    csv_path = custom_data_path or (FULL_CSV_PATH if FULL_CSV_PATH.exists() else DEFAULT_CSV_PATH)
    if not csv_path.exists():
        logger.info(f"No CSV dataset found at {csv_path}. Generating starter sample TWCS dataset...")
        generate_sample_dataset(output_path=DEFAULT_CSV_PATH)
        csv_path = DEFAULT_CSV_PATH

    # Generate Golden Set if not present
    if not GOLDEN_SET_PATH.exists():
        logger.info("Generating 200-sample human-reviewed golden evaluation benchmark...")
        generate_golden_evaluation_set(output_path=GOLDEN_SET_PATH, count=200)

    # Load & Filter
    df_raw = load_raw_data(csv_path)
    brand_df = filter_by_brand(df_raw, brand_name=brand_name)

    # Reconstruct multi-turn conversations
    conversations = build_conversations(brand_df, brand_name=brand_name)
    if not conversations:
        logger.warning(f"No conversation pairs found for brand @{brand_name}. Building on available data.")
        conversations = build_conversations(df_raw, brand_name=brand_name)

    # Persist to SQLite
    save_conversations_to_db(conversations, db_path=DB_PATH)
    logger.info(f"[SUCCESS] Data stage complete. {len(conversations)} conversations persisted to {DB_PATH}")
    return conversations


def stage_train(brand_name: str = DEFAULT_BRAND):
    """Trains TF-IDF baseline, Semantic Classifier, and builds FAISS index."""
    logger.info("=" * 60)
    logger.info(f"STAGE 2: MODEL TRAINING & FAISS INDEXING FOR @{brand_name}")
    logger.info("=" * 60)

    # Fetch conversations from SQLite
    convs = get_conversations_from_db(brand_name=brand_name, db_path=DB_PATH)
    if not convs:
        logger.warning("No brand-filtered conversations in DB. Loading all available conversations.")
        convs = get_conversations_from_db(db_path=DB_PATH)

    if not convs:
        raise ValueError("Database is empty. Please run --stage data first.")

    texts = [c["cleaned_customer_query"] for c in convs if c.get("cleaned_customer_query")]
    labels = [c["intent"] for c in convs if c.get("cleaned_customer_query")]

    # 1. Train TF-IDF Baseline
    logger.info("Training TF-IDF + Logistic Regression Baseline...")
    tfidf_clf = TFIDFIntentClassifier()
    tfidf_clf.fit(texts, labels)
    tfidf_clf.save(TFIDF_MODEL_PATH)

    # 2. Train Semantic Sentence Transformer Classifier
    logger.info("Training Calibrated Semantic Classifier (SentenceTransformers)...")
    semantic_clf = SemanticIntentClassifier()
    semantic_clf.fit(texts, labels)
    semantic_clf.save(SEMANTIC_CLASSIFIER_PATH)

    # 3. Build FAISS Historical Vector Index with Anti-Leakage Partition
    logger.info("Building FAISS semantic retrieval index...")
    retriever = FAISSRetriever()
    retriever.build_index(convs)
    retriever.save(FAISS_INDEX_PATH)

    logger.info("[SUCCESS] Training stage complete. All artifacts saved to models/ directory.")


def stage_eval(brand_name: str = DEFAULT_BRAND):
    """Runs full comparative evaluation on the Golden Set and outputs results."""
    logger.info("=" * 60)
    logger.info(f"STAGE 3: BENCHMARK EVALUATION & AUDITING FOR @{brand_name}")
    logger.info("=" * 60)

    results = run_full_evaluation(
        golden_set_path=GOLDEN_SET_PATH,
        brand_name=brand_name,
        output_path=EVAL_RESULTS_PATH,
    )

    # Print Pretty Summary Tables
    print("\n" + "=" * 70)
    print(f"       BRANDSUPPORT AI — BENCHMARK EVALUATION REPORT (@{brand_name})")
    print("=" * 70)
    
    print("\n[1] INTENT CLASSIFICATION COMPARISON (Test Set N=200):")
    print("-" * 70)
    print(f"{'Model Architecture':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'Status':<10}")
    print("-" * 70)
    for model_name, metrics in results["models_comparison"].items():
        print(f"{model_name:<35} | {metrics['accuracy']:<10.4f} | {metrics['macro_f1']:<10.4f} | {'Evaluated'}")
    print("-" * 70)

    print("\n[2] FAISS RETRIEVAL PERFORMANCE:")
    print("-" * 70)
    ret = results["retrieval_performance"]
    print(f"Recall@1: {ret.get('recall@1', 0):.4f}  |  Recall@3: {ret.get('recall@3', 0):.4f}  |  Recall@5: {ret.get('recall@5', 0):.4f}  |  MRR: {ret.get('mrr', 0):.4f}")
    print("-" * 70)

    print("\n[3] ESCALATION DECISION ENGINE:")
    print("-" * 70)
    esc = results["escalation_performance"]
    print(f"Accuracy: {esc['accuracy']:.4f} | Precision: {esc['precision']:.4f} | Recall: {esc['recall']:.4f} | F1: {esc['f1_score']:.4f}")
    print(f"Safety Penalty Cost: {esc['safety_risk_cost']} (Missed Escalation FN: {esc['false_negatives_missed_escalation']}, Unnecessary FP: {esc['false_positives_unnecessary_escalation']})")
    print("-" * 70)

    print("\n[4] LLM-AS-A-JUDGE QUALITY RUBRIC (Scale 1.0 - 5.0):")
    print("-" * 70)
    judge = results["llm_judge_scores"]
    print(f"Relevance: {judge['mean_relevance']:.2f}/5.0  |  Faithfulness: {judge['mean_faithfulness']:.2f}/5.0  |  Helpfulness: {judge['mean_helpfulness']:.2f}/5.0")
    print(f"Unsupported Claims / Hallucination Rate: {judge['unsupported_claims_rate'] * 100:.1f}%")
    print(f"Escalation Correctness: {judge['mean_escalation_correctness']:.2f}/5.0")
    print("-" * 70)
    print(f"\n[INFO] Full JSON report saved to: {EVAL_RESULTS_PATH}\n")


def list_available_brands(custom_data_path: Path = None):
    """Scans dataset and lists top brand handles with conversation counts."""
    csv_path = custom_data_path or (FULL_CSV_PATH if FULL_CSV_PATH.exists() else DEFAULT_CSV_PATH)
    if not csv_path.exists():
        logger.info(f"Dataset not found at {csv_path}. Generating starter dataset first...")
        generate_sample_dataset(output_path=DEFAULT_CSV_PATH)
        csv_path = DEFAULT_CSV_PATH

    df = load_raw_data(csv_path)
    brands = get_available_brands(df)

    print("\n" + "=" * 50)
    print("      AVAILABLE BRANDS IN DATASET")
    print("=" * 50)
    print(f"{'Brand Handle':<25} | {'Outbound Replies':<15}")
    print("-" * 50)
    if brands:
        for brand, count in brands.items():
            print(f"@{brand:<24} | {count:<15,}")
    else:
        for b in SUPPORTED_BRANDS:
            print(f"@{b:<24} | {'Available'}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="BrandSupport AI — Trustworthy AI Customer Support Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["data", "train", "eval", "all"],
        default="all",
        help="Pipeline stage to execute: 'data' (ingest/build conversations), 'train' (train classifiers & FAISS), 'eval' (evaluate on golden set), or 'all'",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default=DEFAULT_BRAND,
        help="Target brand handle to filter and model (e.g., AppleSupport, AmazonHelp, SpotifyCares)",
    )
    parser.add_argument(
        "--list-brands",
        action="store_true",
        help="List all brand handles available in the Twitter Customer Support dataset",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional custom path to CSV dataset file (defaults to data/sample_twcs.csv or data/twcs.csv)",
    )

    args = parser.parse_args()

    if args.list_brands:
        custom_path = Path(args.data_path) if args.data_path else None
        list_available_brands(custom_path)
        return

    custom_path = Path(args.data_path) if args.data_path else None

    if args.stage in ["data", "all"]:
        stage_data(brand_name=args.brand, custom_data_path=custom_path)

    if args.stage in ["train", "all"]:
        stage_train(brand_name=args.brand)

    if args.stage in ["eval", "all"]:
        stage_eval(brand_name=args.brand)


if __name__ == "__main__":
    main()
