"""Benchmark the existing CHROM rPPG pipeline against UBFC-rPPG ground truth.

Ground-truth format (3 × N array):
  row 0 — BVP waveform (normalised)
  row 1 — HR in bpm (per-frame, from the CMS50E pulse oximeter)
  row 2 — timestamps (seconds)

Usage:
    python benchmark_rppg.py --archive e:/ai-intern/archive
    python benchmark_rppg.py --archive e:/ai-intern/archive --subject subject1

Outputs a summary table + overall MAE and RMSE.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make project root importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from services.video_analysis.rppg import analyze_rppg_from_video


# ── Ground-truth helpers ──────────────────────────────────────────────────────

def load_ground_truth(gt_path: str):
    """Return (mean_hr_bpm, bvp_signal, timestamps) from UBFC ground-truth file."""
    data = np.loadtxt(gt_path)  # shape (3, N)
    bvp = data[0]
    hr  = data[1]   # per-frame HR from pulse-ox
    ts  = data[2]
    return float(np.mean(hr)), bvp, ts


def bvp_to_hr(bvp: np.ndarray, timestamps: np.ndarray) -> float:
    """Derive mean HR from the BVP waveform using peak detection (scipy)."""
    try:
        from scipy.signal import find_peaks
        fps_gt = 1.0 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 30.0
        peaks, _ = find_peaks(bvp, distance=int(fps_gt * 0.4))
        if len(peaks) < 2:
            return float("nan")
        rr_s = np.diff(peaks) / fps_gt
        return float(60.0 / np.mean(rr_s))
    except Exception:
        return float("nan")


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(archive_dir: str, subject_filter=None):
    archive = Path(archive_dir)
    subjects = sorted(d for d in archive.iterdir() if d.is_dir())

    if subject_filter:
        subjects = [s for s in subjects if s.name == subject_filter]

    if not subjects:
        print(f"No subjects found in {archive_dir}")
        return

    print(f"\n{'Subject':<12} {'GT HR':>8} {'Est HR':>8} {'Error':>8} {'HRV(ms)':>9} {'Status'}")
    print("-" * 60)

    errors = []
    successes = 0
    failures = 0

    for subj in subjects:
        vid  = subj / "vid.avi"
        gt   = subj / "ground_truth.txt"

        if not vid.exists() or not gt.exists():
            continue

        gt_hr, bvp, ts = load_ground_truth(str(gt))

        result = analyze_rppg_from_video(str(vid))

        if not result["data_available"]:
            print(f"{subj.name:<12} {gt_hr:>8.1f} {'—':>8} {'—':>8} {'—':>9}  FAIL (no data)")
            failures += 1
            continue

        est_hr  = result["hr_bpm"]
        rmssd   = result["avg_hrv_rmssd"]
        err     = est_hr - gt_hr
        errors.append(abs(err))
        successes += 1

        rmssd_str = f"{rmssd:.1f}" if rmssd is not None else "—"
        print(
            f"{subj.name:<12} {gt_hr:>8.1f} {est_hr:>8.1f} {err:>+8.1f} "
            f"{rmssd_str:>9}  OK"
        )

    print("-" * 60)
    total = successes + failures
    print(f"\nResults: {successes}/{total} subjects processed successfully")
    if errors:
        mae  = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        print(f"  MAE  (mean absolute error):  {mae:.2f} bpm")
        print(f"  RMSE (root mean sq error):   {rmse:.2f} bpm")
        print()
        if mae < 5:
            print("  Assessment: GOOD — CHROM is performing well on UBFC data.")
        elif mae < 10:
            print("  Assessment: MODERATE — CHROM needs tuning or a DL upgrade.")
        else:
            print("  Assessment: POOR — Consider replacing CHROM with a trained model.")
    else:
        print("  No successful measurements to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark rPPG vs UBFC-rPPG ground truth")
    parser.add_argument("--archive", default="e:/ai-intern/archive", help="Path to archive folder")
    parser.add_argument("--subject", default=None, help="Run on a single subject (e.g. subject1)")
    args = parser.parse_args()

    run_benchmark(args.archive, args.subject)
