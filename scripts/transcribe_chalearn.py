"""
Step 1: Transcribe ChaLearn First Impressions videos → transcripts.csv

Usage:
    python scripts/transcribe_chalearn.py --split train --model base
    python scripts/transcribe_chalearn.py --split val --model base

Output:
    data/transcripts_train.csv  (columns: video_id, transcript)
    data/transcripts_val.csv

Resume-safe: skips already-transcribed videos.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "first-impressions"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


ANNOTATION_PATHS = {
    "train": BASE_DIR / "data" / "annotations" / "annotation_training.pkl",
    "val":   BASE_DIR / "data" / "annotations" / "annotation_validation.pkl",
    "test":  BASE_DIR / "data" / "annotations" / "annotation_test.pkl",
}


def load_annotations(split: str) -> dict:
    """Load pickle annotations for a split."""
    import pickle

    pkl_path = ANNOTATION_PATHS.get(split)
    if not pkl_path or not pkl_path.exists():
        print(f"[warn] annotations for '{split}' not found — skipping label load")
        return {}
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    video_ids = list(raw["openness"].keys())
    labels = {}
    for vid in video_ids:
        labels[vid] = {t: raw[t][vid] for t in TRAITS}
        labels[vid]["interview"] = raw["interview"][vid]
    return labels


def get_video_paths(split: str) -> list:
    split_dir = DATASET_DIR / split
    if not split_dir.exists():
        sys.exit(f"[error] split directory not found: {split_dir}")
    videos = sorted(split_dir.glob("*.mp4"))
    print(f"[info] found {len(videos)} videos in {split_dir}")
    return videos


def transcribe_videos(
    videos: list,
    output_csv: Path,
    whisper_model: str = "base",
) -> None:
    import whisper

    # load already-done video ids
    done: set[str] = set()
    if output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("transcript", "").strip():
                    done.add(row["video_id"])
        print(f"[info] resuming — {len(done)} already transcribed")

    model = whisper.load_model(whisper_model)
    print(f"[info] loaded whisper model: {whisper_model}")

    # open csv in append mode
    file_exists = output_csv.exists()
    f_out = open(output_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=["video_id", "transcript"])
    if not file_exists:
        writer.writeheader()

    total = len(videos)
    for i, video_path in enumerate(videos):
        vid_id = video_path.name
        if vid_id in done:
            continue

        t0 = time.time()
        try:
            result = model.transcribe(str(video_path), language="en", fp16=False)
            transcript = result["text"].strip()
        except Exception as e:
            print(f"[warn] failed {vid_id}: {e}")
            transcript = ""

        writer.writerow({"video_id": vid_id, "transcript": transcript})
        f_out.flush()

        elapsed = time.time() - t0
        print(f"[{i+1}/{total}] {vid_id} ({elapsed:.1f}s): {transcript[:80]!r}")

    f_out.close()
    print(f"\n[done] saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe ChaLearn videos with Whisper")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (tiny=fastest, large=most accurate)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit to first N videos (0 = all)"
    )
    args = parser.parse_args()

    output_csv = OUTPUT_DIR / f"transcripts_{args.split}.csv"
    videos = get_video_paths(args.split)

    if args.limit > 0:
        videos = videos[: args.limit]
        print(f"[info] limited to {len(videos)} videos")

    transcribe_videos(videos, output_csv, whisper_model=args.model)


if __name__ == "__main__":
    main()
