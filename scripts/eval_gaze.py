"""
eval_gaze.py — Standalone GazeCapture evaluation for affine gaze calibration.

Uses OpenCV Haar cascades (no downloads needed) to detect eye centres, then
evaluates your affine-calibration approach against GazeCapture ground truth.

Compares:
  personalized — affine transform fitted per-session (your system)
  fixed        — raw eye-centre coordinates, no calibration (baseline)

Usage:
    python eval_gaze.py
    python scripts/eval_gaze.py --sessions 20 --output data/gaze_eval.json
"""

import sys
import json
import argparse
import traceback
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List


# ── Haar cascades (built into opencv-python, no download needed) ──────────────
_CV2_DATA   = Path(cv2.__file__).parent / "data"
FACE_CASCADE = cv2.CascadeClassifier(str(_CV2_DATA / "haarcascade_frontalface_default.xml"))
EYE_CASCADE  = cv2.CascadeClassifier(str(_CV2_DATA / "haarcascade_eye.xml"))

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
N_CALIB     = 15   # calibration frames (matches your 15-point system)
MIN_TEST    = 10   # skip session if fewer usable test frames

NEURODIVERSITY_VARIANCE_THRESHOLD = 0.06
NEURODIVERSITY_SCALE              = 1.4


# ── Affine helpers (mirrors calibration_runner logic) ─────────────────────────

def _fit_affine(iris: np.ndarray, screen: np.ndarray) -> np.ndarray:
    """Fit  screen ≈ [ex, ey, 1] @ A  via least-squares. Returns A (3×2)."""
    N  = iris.shape[0]
    ih = np.hstack([iris, np.ones((N, 1))])
    A, _, _, _ = np.linalg.lstsq(ih, screen, rcond=None)
    return A


def _apply(eye_xy: tuple, A: np.ndarray) -> tuple:
    """Map raw eye centre through 3×2 affine A."""
    v   = np.array([eye_xy[0], eye_xy[1], 1.0])
    out = v @ A
    return (float(out[0]), float(out[1]))


# ── Eye-centre extraction ─────────────────────────────────────────────────────

def get_eye_centre(image_path: Path) -> Optional[tuple]:
    """
    Detect face → find eyes inside face ROI → return normalised (x, y) of
    average eye centre relative to the full image.
    Returns None if face or eyes not found.
    """
    img  = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None

    # Use largest face
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    roi_gray = gray[fy:fy+fh, fx:fx+fw]

    eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
    if len(eyes) == 0:
        return None

    # Average centre of all detected eyes (normalised to full image)
    cx = float(np.mean([fx + ex + ew / 2.0 for ex, ey, ew, eh in eyes])) / w
    cy = float(np.mean([fy + ey + eh / 2.0 for ex, ey, ew, eh in eyes])) / h
    return (cx, cy)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_session(session_path: Path) -> Optional[Dict]:
    """
    Load one GazeCapture session.
    dotInfo.XPts/YPts = screen pixel coords of the dot shown to subject.
    screen.W/H        = screen size in pixels (per frame, orientation-aware).
    frames.json       = list of frame filenames.
    """
    dot_path    = session_path / "dotInfo.json"
    screen_path = session_path / "screen.json"
    frames_path = session_path / "frames.json"
    frames_dir  = session_path / "frames"

    if not all(p.exists() for p in [dot_path, screen_path, frames_path, frames_dir]):
        return None

    with open(dot_path)    as f: dot    = json.load(f)
    with open(screen_path) as f: screen = json.load(f)
    with open(frames_path) as f: frames = json.load(f)

    xpts = np.array(dot["XPts"], dtype=float)
    ypts = np.array(dot["YPts"], dtype=float)
    ws   = np.array(screen["W"], dtype=float)
    hs   = np.array(screen["H"], dtype=float)

    n = min(len(xpts), len(ypts), len(ws), len(hs), len(frames))
    if n < N_CALIB + MIN_TEST:
        return None

    norm_x = xpts[:n] / ws[:n]
    norm_y = ypts[:n] / hs[:n]

    return {
        "frames_dir" : frames_dir,
        "frame_names": frames[:n],
        "norm_x"     : norm_x,
        "norm_y"     : norm_y,
        "n"          : n,
    }


# ── Per-session evaluation ────────────────────────────────────────────────────

def evaluate_session(session: Dict) -> Optional[Dict]:
    """
    Frames 0…N_CALIB-1  → calibration set (fit affine transform).
    Frames N_CALIB…end  → test set (measure MAE).
    """
    frames_dir  = session["frames_dir"]
    frame_names = session["frame_names"]
    norm_x      = session["norm_x"]
    norm_y      = session["norm_y"]

    # ── Step 1: build calibration measurements ────────────────────────────────
    measurements = []
    for i in range(N_CALIB):
        eye = get_eye_centre(frames_dir / frame_names[i])
        if eye is None:
            continue
        measurements.append({
            "ex": eye[0], "ey": eye[1],
            "sx": float(norm_x[i]), "sy": float(norm_y[i]),
        })

    if len(measurements) < 6:
        return None

    # ── Step 2: fit affine transform ──────────────────────────────────────────
    iris_arr   = np.array([[m["ex"], m["ey"]] for m in measurements])
    screen_arr = np.array([[m["sx"], m["sy"]] for m in measurements])
    A = _fit_affine(iris_arr, screen_arr)

    # Baseline variance across calibration eye positions
    diffs = np.diff(iris_arr, axis=0)
    baseline_var = float(np.mean(np.sqrt((diffs**2).sum(axis=1))))
    neuro_adj    = NEURODIVERSITY_SCALE if baseline_var > NEURODIVERSITY_VARIANCE_THRESHOLD else 1.0
    calib_quality = max(0.0, 1.0 - baseline_var / 0.04)

    # ── Step 3: evaluate on test frames ──────────────────────────────────────
    errors_personalized = []
    errors_fixed        = []

    for i in range(N_CALIB, session["n"]):
        eye = get_eye_centre(frames_dir / frame_names[i])
        if eye is None:
            continue

        gt_x, gt_y     = float(norm_x[i]), float(norm_y[i])
        pred_x, pred_y = _apply(eye, A)

        errors_personalized.append(np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2))
        errors_fixed.append(       np.sqrt((eye[0]  - gt_x)**2 + (eye[1]  - gt_y)**2))

    if len(errors_personalized) < MIN_TEST:
        return None

    return {
        "n_calib"          : len(measurements),
        "n_test"           : len(errors_personalized),
        "baseline_variance": round(baseline_var,   6),
        "neuro_adjustment" : neuro_adj,
        "calib_quality"    : round(calib_quality,  4),
        "mae_personalized" : round(float(np.mean(errors_personalized)), 6),
        "mae_fixed"        : round(float(np.mean(errors_fixed)),        6),
        "std_personalized" : round(float(np.std(errors_personalized)),  6),
        "std_fixed"        : round(float(np.std(errors_fixed)),         6),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="dataset",                help="Dataset root")
    parser.add_argument("--sessions", type=int, default=10,             help="Max sessions")
    parser.add_argument("--output",   default="data/gaze_eval.json", help="Output JSON")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset not found: {dataset_root.resolve()}")
        sys.exit(1)

    # Collect session paths  (structure: dataset/{id}/{id}/)
    session_paths: List[Path] = []
    for outer in sorted(dataset_root.iterdir()):
        if not outer.is_dir():
            continue
        for inner in sorted(outer.iterdir()):
            if inner.is_dir() and (inner / "dotInfo.json").exists():
                session_paths.append(inner)
        if (outer / "dotInfo.json").exists():
            session_paths.append(outer)

    session_paths = session_paths[:args.sessions]
    print(f"[INFO] Found {len(session_paths)} sessions\n")

    results = []
    for idx, path in enumerate(session_paths):
        print(f"[{idx+1:3d}/{len(session_paths)}] {path.name} ...", end=" ", flush=True)
        try:
            session = load_session(path)
            if session is None:
                print("SKIP (short/missing)")
                continue
            res = evaluate_session(session)
            if res is None:
                print("SKIP (too few face detections)")
                continue
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            continue

        res["session"] = path.name
        results.append(res)
        delta = res["mae_fixed"] - res["mae_personalized"]
        tag   = " [neuro]" if res["neuro_adjustment"] > 1.0 else ""
        print(f"MAE_cal={res['mae_personalized']:.4f}  MAE_raw={res['mae_fixed']:.4f}  D={delta:+.4f}{tag}")

    if not results:
        print("\n[ERROR] No sessions produced results.")
        sys.exit(1)

    mae_p = [r["mae_personalized"] for r in results]
    mae_f = [r["mae_fixed"]        for r in results]
    neuro = [r for r in results if r["neuro_adjustment"] > 1.0]

    summary = {
        "sessions_evaluated"      : len(results),
        "overall_mae_personalized": round(float(np.mean(mae_p)), 6),
        "overall_mae_fixed"       : round(float(np.mean(mae_f)), 6),
        "std_personalized"        : round(float(np.std(mae_p)),  6),
        "std_fixed"               : round(float(np.std(mae_f)),  6),
        "improvement_over_fixed"  : round(float(np.mean(mae_f)) - float(np.mean(mae_p)), 6),
        "neurodiversity_sessions" : len(neuro),
        "session_results"         : results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 56)
    print("RESULTS")
    print("=" * 56)
    print(f"Sessions evaluated     : {summary['sessions_evaluated']}")
    print(f"MAE personalized cal   : {summary['overall_mae_personalized']:.4f} +/-{summary['std_personalized']:.4f}")
    print(f"MAE raw (no cal)       : {summary['overall_mae_fixed']:.4f} +/-{summary['std_fixed']:.4f}")
    print(f"Improvement            : {summary['improvement_over_fixed']:+.4f}  (negative = your system wins)")
    print(f"High-variance sessions : {summary['neurodiversity_sessions']}/{summary['sessions_evaluated']}")
    print(f"Results saved to       : {out.resolve()}")
    print("=" * 56)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
