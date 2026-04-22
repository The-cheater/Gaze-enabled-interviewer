"""
Step 2: Train an OCEAN regression model from transcripts + ChaLearn labels.

Pipeline:
    transcript → sentence-transformer embeddings (768-dim)
               → MLP regression (5 heads: O C E A N)
               → trained model saved to models/ocean_regressor.pkl

Usage:
    python scripts/train_ocean_regressor.py
    python scripts/train_ocean_regressor.py --epochs 50 --batch 64

Requirements:
    pip install sentence-transformers scikit-learn numpy pandas

Outputs:
    models/ocean_regressor.pkl   — trained sklearn Pipeline (scaler + MLP)
    models/ocean_encoder.txt     — sentence-transformer model name used
    data/train_embeddings.npy    — cached embeddings (skip re-encoding on rerun)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
ANNOTATIONS = {
    "train": BASE_DIR / "data" / "annotations" / "annotation_training.pkl",
    "val":   BASE_DIR / "data" / "annotations" / "annotation_validation.pkl",
    "test":  BASE_DIR / "data" / "annotations" / "annotation_test.pkl",
}

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
ENCODER_MODEL = "all-mpnet-base-v2"  # best quality/speed tradeoff


class MultiRidge:
    """Wraps one Ridge pipeline per trait; predict() returns (n_samples, n_traits)."""
    def __init__(self, estimators):
        self.estimators_ = estimators

    def predict(self, X):
        return np.column_stack([e.predict(X) for e in self.estimators_])

# Ridge alpha candidates (log-spaced) — cross-validated automatically
RIDGE_ALPHAS = [1000.0, 10000.0, 50000.0, 100000.0, 500000.0]



# ── data loading ───────────────────────────────────────────────────────────────

def load_labels(split: str = "train") -> pd.DataFrame:
    pkl_path = ANNOTATIONS[split]
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    records = []
    for vid in raw["openness"]:
        row = {"video_id": vid}
        for t in TRAITS:
            row[t] = raw[t][vid]
        records.append(row)
    df = pd.DataFrame(records)
    print(f"[info] loaded {len(df)} {split} labels")
    return df


def load_transcripts(split: str = "train") -> pd.DataFrame:
    csv_path = DATA_DIR / f"transcripts_{split}.csv"
    if not csv_path.exists():
        sys.exit(
            f"[error] {csv_path} not found. Run transcribe_chalearn.py first.\n"
            f"  python scripts/transcribe_chalearn.py --split {split}"
        )
    df = pd.read_csv(csv_path)
    df["transcript"] = df["transcript"].fillna("").str.strip()
    print(f"[info] loaded {len(df)} transcripts from {csv_path}")
    return df


def merge_data(transcripts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transcripts.merge(labels, on="video_id", how="inner")
    # drop empty transcripts
    before = len(df)
    df = df[df["transcript"].str.len() > 5].reset_index(drop=True)
    print(f"[info] merged: {before} → {len(df)} rows (dropped empty transcripts)")
    return df


# ── embeddings ─────────────────────────────────────────────────────────────────

def encode_transcripts(texts: list, cache_path=None) -> np.ndarray:
    if cache_path and cache_path.exists():
        print(f"[info] loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    print(f"[info] encoding {len(texts)} transcripts with {ENCODER_MODEL}...")
    model = SentenceTransformer(ENCODER_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    if cache_path:
        np.save(cache_path, embeddings)
        print(f"[info] cached embeddings → {cache_path}")
    return embeddings


# ── model training ─────────────────────────────────────────────────────────────

def train(X: np.ndarray, Y: np.ndarray, args) -> object:
    """
    Train one RidgeCV per trait.

    Why Ridge instead of MLP:
    - ChaLearn personality labels are noisy (low inter-rater agreement)
    - 768-dim embeddings + ~6k samples → MLPs overfit badly (negative val R²)
    - Ridge with cross-validated alpha is the standard ChaLearn baseline
      and consistently outperforms deep models on text-only features
    """
    import copy
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    print(f"\n[info] training RidgeCV on X={X.shape}, Y={Y.shape}")
    print(f"[info] alpha candidates: {RIDGE_ALPHAS} (5-fold CV per trait)")

    estimators = []
    with tqdm(TRAITS, desc="Training traits", unit="trait") as bar:
        for i, trait in enumerate(bar):
            bar.set_postfix(trait=trait)
            ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=5, scoring="r2")
            pipe = Pipeline([("scaler", StandardScaler()), ("ridge", ridge)])
            pipe.fit(X, Y[:, i])
            best_alpha = pipe.named_steps["ridge"].alpha_
            best_cv = pipe.named_steps["ridge"].best_score_
            tqdm.write(f"  {trait:<22} best_alpha={best_alpha:.1f}  cv_r²={best_cv:.4f}")
            estimators.append(pipe)

    return MultiRidge(estimators)


def evaluate_pearson(model, X: np.ndarray, Y: np.ndarray, split_name: str = "val"):
    from scipy.stats import pearsonr
    from tqdm import tqdm

    print(f"\n[info] running predictions on {len(X)} samples...")
    Y_pred = model.predict(X)
    print(f"\n{'─'*50}")
    print(f"Pearson Correlation — {split_name}")
    print(f"{'─'*50}")
    total = 0.0
    for i, trait in enumerate(tqdm(TRAITS, desc="Pearson r", unit="trait")):
        r, p = pearsonr(Y[:, i], Y_pred[:, i])
        tqdm.write(f"  {trait:<20} r = {r:.4f}  (p={p:.2e})")
        total += r
    mean_r = total / len(TRAITS)
    print(f"{'─'*50}")
    print(f"  {'Mean r':<20} r = {mean_r:.4f}")
    print(f"{'─'*50}\n")
    return mean_r


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split", action="store_true", help="Evaluate on val transcripts too")
    parser.add_argument("--no-cache", action="store_true", help="Re-encode even if cache exists")
    args = parser.parse_args()

    # 1. load data
    train_labels = load_labels("train")
    train_transcripts = load_transcripts("train")
    df = merge_data(train_transcripts, train_labels)

    texts = df["transcript"].tolist()
    Y = df[TRAITS].values.astype(np.float32)

    # 2. encode train
    cache = DATA_DIR / "train_embeddings.npy" if not args.no_cache else None
    X = encode_transcripts(texts, cache_path=cache)

    # 3. train on full train set (real val set is now available)
    print(f"[info] training on {len(X)} samples (full train set)")
    model = train(X, Y, args)

    # 4. evaluate — use real val labels if available, else internal 10% split
    val_pkl_exists = ANNOTATIONS["val"].exists()
    val_csv = DATA_DIR / "transcripts_val.csv"

    if val_csv.exists() and val_pkl_exists:
        val_labels = load_labels("val")
        val_transcripts = load_transcripts("val")
        val_df = merge_data(val_transcripts, val_labels)
        val_texts = val_df["transcript"].tolist()
        val_Y = val_df[TRAITS].values.astype(np.float32)
        val_cache = DATA_DIR / "val_embeddings.npy" if not args.no_cache else None
        val_X = encode_transcripts(val_texts, cache_path=val_cache)
        evaluate_pearson(model, val_X, val_Y, split_name="val set (official ChaLearn val)")
        val_pred = model.predict(val_X)
        out_df = val_df[["video_id"]].copy()
        for i, t in enumerate(TRAITS):
            out_df[t] = val_pred[:, i]
        out_df.to_csv(DATA_DIR / "val_predictions.csv", index=False)
        print(f"[saved] val predictions → {DATA_DIR / 'val_predictions.csv'}")
    else:
        # fallback: internal 10% split from train
        from sklearn.model_selection import train_test_split
        X_tr, X_hv, Y_tr, Y_hv = train_test_split(X, Y, test_size=0.1, random_state=42)
        model_hv = train(X_tr, Y_tr, args)
        evaluate_pearson(model_hv, X_hv, Y_hv, split_name="internal_val (10% held-out from train)")
        if not val_pkl_exists:
            print("[hint] place annotation_validation.pkl in E:/ai-intern/ for official val eval")

    # 5. save model
    model_path = MODELS_DIR / "ocean_regressor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "traits": TRAITS, "encoder": ENCODER_MODEL}, f)
    print(f"[saved] model → {model_path}")
    (MODELS_DIR / "ocean_encoder.txt").write_text(ENCODER_MODEL)

    # 6. optional test set evaluation
    if args.val_split:
        test_csv = DATA_DIR / "transcripts_test.csv"
        if test_csv.exists():
            test_labels = load_labels("test")
            test_transcripts = load_transcripts("test")
            test_df = merge_data(test_transcripts, test_labels)
            test_X = encode_transcripts(
                test_df["transcript"].tolist(),
                cache_path=DATA_DIR / "test_embeddings.npy" if not args.no_cache else None,
            )
            test_Y = test_df[TRAITS].values.astype(np.float32)
            evaluate_pearson(model, test_X, test_Y, split_name="test set (official ChaLearn test)")


if __name__ == "__main__":
    main()
