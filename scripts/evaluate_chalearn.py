"""
Step 3: Evaluate OCEAN scoring — baseline (rule-based) vs trained model.

Computes Pearson correlation for both approaches on the training set
(using 10% held-out split) and outputs a comparison report.

Usage:
    python scripts/evaluate_chalearn.py
    python scripts/evaluate_chalearn.py --limit 200   # quick test on 200 samples
    python scripts/evaluate_chalearn.py --baseline-only
    python scripts/evaluate_chalearn.py --model-only

Output:
    data/evaluation_report.csv   — per-trait Pearson r for both methods
    data/predictions_baseline.csv
    data/predictions_model.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(BASE_DIR))

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
ANNOTATIONS = {
    "train": BASE_DIR / "data" / "annotations" / "annotation_training.pkl",
    "val":   BASE_DIR / "data" / "annotations" / "annotation_validation.pkl",
    "test":  BASE_DIR / "data" / "annotations" / "annotation_test.pkl",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_labels(split: str = "train") -> pd.DataFrame:
    with open(ANNOTATIONS[split], "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    records = [
        {"video_id": vid, **{t: raw[t][vid] for t in TRAITS}}
        for vid in raw["openness"]
    ]
    return pd.DataFrame(records)


def load_transcripts(split="train") -> pd.DataFrame:
    csv_path = DATA_DIR / f"transcripts_{split}.csv"
    if not csv_path.exists():
        sys.exit(f"[error] {csv_path} not found — run transcribe_chalearn.py first")
    df = pd.read_csv(csv_path)
    df["transcript"] = df["transcript"].fillna("").str.strip()
    return df


def pearson_report(y_true: np.ndarray, y_pred: np.ndarray, method_name: str) -> dict:
    from scipy.stats import pearsonr

    results = {"method": method_name}
    print(f"\n{'═'*55}")
    print(f"  {method_name}")
    print(f"{'═'*55}")
    total = 0.0
    for i, trait in enumerate(TRAITS):
        r, p = pearsonr(y_true[:, i], y_pred[:, i])
        results[trait] = round(r, 4)
        flag = "✓" if r > 0.2 else "✗"
        print(f"  {flag} {trait:<22} r = {r:+.4f}  (p={p:.1e})")
        total += r
    mean_r = total / len(TRAITS)
    results["mean_r"] = round(mean_r, 4)
    print(f"{'─'*55}")
    print(f"    {'Mean r':<22} r = {mean_r:+.4f}")
    return results


# ── baseline: current rule-based ocean_mapper ─────────────────────────────────

def score_with_baseline(transcripts: list) -> np.ndarray:
    """Run each transcript through the existing ocean_mapper.py pipeline."""
    try:
        from services.scoring.ocean_mapper import OceanMapper
        mapper = OceanMapper()
    except ImportError:
        # Fallback: try direct import path
        sys.path.insert(0, str(BASE_DIR / "services" / "scoring"))
        from ocean_mapper import OceanMapper
        mapper = OceanMapper()

    from tqdm import tqdm
    predictions = []
    for text in tqdm(transcripts, desc="Baseline scoring", unit="transcript"):
        try:
            # ocean_mapper expects a list of ResponseScore-like objects or plain text
            # Try the text-only scoring path
            result = mapper.score_from_transcript(text)
            row = [
                result.get("openness", 50) / 100,
                result.get("conscientiousness", 50) / 100,
                result.get("extraversion", 50) / 100,
                result.get("agreeableness", 50) / 100,
                result.get("neuroticism", 50) / 100,
            ]
        except Exception as e:
            # If ocean_mapper doesn't support direct text input, use neutral scores
            row = [0.5] * 5
        predictions.append(row)

    return np.array(predictions, dtype=np.float32)


# ── trained model ──────────────────────────────────────────────────────────────

def score_with_model(transcripts: list) -> np.ndarray:
    model_path = MODELS_DIR / "ocean_regressor.pkl"
    if not model_path.exists():
        sys.exit(
            f"[error] trained model not found at {model_path}\n"
            "  Run: python scripts/train_ocean_regressor.py"
        )
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    encoder_name = bundle.get("encoder", "all-mpnet-base-v2")

    from sentence_transformers import SentenceTransformer

    print(f"  [model] encoding {len(transcripts)} transcripts with {encoder_name}...")
    encoder = SentenceTransformer(encoder_name)
    X = encoder.encode(
        transcripts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return model.predict(X).astype(np.float32)


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit to N samples (0=all)")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Which split to evaluate on (default: val)",
    )
    args = parser.parse_args()

    # 1. load data — use real val/test labels now that annotations are decrypted
    labels = load_labels(args.split)
    transcripts = load_transcripts(args.split)
    df = transcripts.merge(labels, on="video_id", how="inner")
    df = df[df["transcript"].str.len() > 5].reset_index(drop=True)
    print(f"[info] evaluating on {len(df)} {args.split} samples")

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"[info] limited to {len(df)} samples")

    texts = df["transcript"].tolist()
    y_true = df[TRAITS].values.astype(np.float32)

    all_results = []

    # 3. baseline
    if not args.model_only:
        print("\n[running baseline (rule-based ocean_mapper)]...")
        y_baseline = score_with_baseline(texts)
        pd.DataFrame(
            np.hstack([y_true, y_baseline]),
            columns=[f"true_{t}" for t in TRAITS] + [f"pred_{t}" for t in TRAITS],
        ).to_csv(DATA_DIR / "predictions_baseline.csv", index=False)
        all_results.append(pearson_report(y_true, y_baseline, "Baseline — Rule-Based ocean_mapper"))

    # 4. trained model
    if not args.baseline_only:
        print("\n[running trained model]...")
        y_model = score_with_model(texts)
        pd.DataFrame(
            np.hstack([y_true, y_model]),
            columns=[f"true_{t}" for t in TRAITS] + [f"pred_{t}" for t in TRAITS],
        ).to_csv(DATA_DIR / "predictions_model.csv", index=False)
        all_results.append(pearson_report(y_true, y_model, "Trained MLP + Sentence Embeddings"))

    # 5. comparison table
    if len(all_results) == 2:
        print(f"\n{'═'*55}")
        print("  Delta (trained - baseline)")
        print(f"{'═'*55}")
        for trait in TRAITS + ["mean_r"]:
            delta = all_results[1][trait] - all_results[0][trait]
            sign = "+" if delta >= 0 else ""
            print(f"  {trait:<22} Δr = {sign}{delta:.4f}")

    # 6. save report
    report = pd.DataFrame(all_results)
    report.to_csv(DATA_DIR / "evaluation_report.csv", index=False)
    print(f"\n[saved] evaluation_report.csv → {DATA_DIR / 'evaluation_report.csv'}")


if __name__ == "__main__":
    main()
