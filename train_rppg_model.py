"""Train a spectral MLP to estimate HR from CHROM pulse signal.

The model learns to select the correct fundamental cardiac frequency from the
power spectral density of the CHROM signal, fixing the sub-harmonic locking
problem that causes CHROM's FFT peak-picker to land on 0.5x or 2x the true HR.

Architecture: MLP on a 64-bin normalised PSD interpolated onto a fixed
frequency grid (0.75–3 Hz), so it is sample-rate agnostic at inference time.

Training data: UBFC-rPPG archive/ folder (42 subjects).
  archive/subjectN/vid.avi          — face video (30 fps)
  archive/subjectN/ground_truth.txt — (3 × T): BVP, HR(bpm), timestamps

Output: services/video_analysis/rppg_model.pt

Usage:
    python train_rppg_model.py --archive e:/ai-intern/archive
    python train_rppg_model.py --archive e:/ai-intern/archive --epochs 300
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────

N_BINS      = 64           # fixed frequency grid bins (fps-agnostic)
FREQ_LOW    = 0.75         # cardiac band lower bound (Hz)
FREQ_HIGH   = 3.0          # cardiac band upper bound (Hz)
WIN_SEC     = 10.0         # window length in seconds
STRIDE_SEC  = 5.0          # window stride (overlapping windows → more samples)
MIN_WIN_SEC = 5.0          # minimum window length to include

FREQ_GRID = np.linspace(FREQ_LOW, FREQ_HIGH, N_BINS)  # target frequency axis

MODEL_PATH = Path(__file__).parent / "services" / "video_analysis" / "rppg_model.pt"

# ── CHROM signal extraction (mirrors rppg.py) ─────────────────────────────────

def _butter_bandpass(fps: float, order: int = 4):
    from scipy.signal import butter
    nyq = fps / 2.0
    low  = max(1e-3, min(FREQ_LOW  / nyq, 0.999))
    high = max(1e-3, min(FREQ_HIGH / nyq, 0.999))
    if low >= high:
        return None
    return butter(order, [low, high], btype="band")


def _apply_bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    coeffs = _butter_bandpass(fps)
    if coeffs is not None:
        try:
            from scipy.signal import filtfilt
            b, a = coeffs
            return filtfilt(b, a, signal)
        except Exception:
            pass
    N = len(signal)
    freqs = np.fft.fftfreq(N, d=1.0 / fps)
    F = np.fft.fft(signal)
    mask = (np.abs(freqs) < FREQ_LOW) | (np.abs(freqs) > FREQ_HIGH)
    F[mask] = 0
    return np.real(np.fft.ifft(F))


def chrom_signal(rgb: np.ndarray, fps: float) -> np.ndarray:
    """CHROM decomposition → bandpass-filtered pulse signal."""
    means = rgb.mean(axis=0)
    means = np.where(means == 0, 1e-6, means)
    norm  = rgb / means
    R, G, B = norm[:, 0], norm[:, 1], norm[:, 2]
    Xs = 3 * R - 2 * G
    Ys = 1.5 * R + G - 1.5 * B
    alpha = Xs.std() / (Ys.std() or 1e-6)
    S = Xs - alpha * Ys
    return _apply_bandpass(S, fps)


def psd_features(pulse: np.ndarray, fps: float) -> np.ndarray:
    """Compute normalised PSD interpolated onto FREQ_GRID (fps-agnostic).

    Returns a 64-dim vector summing to 1, where each element corresponds
    to one frequency bin in FREQ_GRID (0.75–3 Hz).
    """
    N     = len(pulse)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    psd   = np.abs(np.fft.rfft(pulse)) ** 2

    # Interpolate onto fixed grid
    feat = np.interp(FREQ_GRID, freqs, psd, left=0.0, right=0.0)

    # Normalise to sum = 1 (scale-invariant)
    total = feat.sum()
    if total > 1e-12:
        feat /= total
    return feat.astype(np.float32)


# ── Face RGB extraction (same Haar + forehead approach as rppg.py) ────────────

def extract_rgb_from_video(video_path: str):
    """Return (rgb_array (T×3), fps) from video, skipping high-motion frames."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = max(1.0, min(raw_fps, 60.0))

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    rgb_list = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            delta = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean())
            if delta > 12.0:
                prev_gray = gray
                continue
        prev_gray = gray

        h, w = frame.shape[:2]
        roi = None

        if not cascade.empty():
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                minSize=(int(w * 0.1), int(h * 0.1)),
            )
            if len(faces) > 0:
                x, y, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                forehead_h = max(1, int(fh * 0.35))
                roi = frame[y: y + forehead_h, x: x + fw]

        if roi is None or roi.size == 0:
            roi = frame[int(h*0.20): int(h*0.55), int(w*0.25): int(w*0.75)]
        if roi.size == 0:
            continue

        bgr = roi.reshape(-1, 3).mean(axis=0)
        rgb_list.append(bgr[::-1].copy())

    cap.release()
    if not rgb_list:
        return None, None
    return np.array(rgb_list, dtype=np.float64), fps


# ── Ground-truth loader ───────────────────────────────────────────────────────

def load_ground_truth(gt_path: str):
    """Return (hr_per_frame, timestamps) from UBFC ground-truth file."""
    data = np.loadtxt(gt_path)  # shape (3, N)
    return data[1], data[2]  # HR(bpm) array, timestamps(s)


def gt_hr_for_window(hr_frames: np.ndarray, timestamps: np.ndarray,
                     t_start: float, t_end: float) -> float:
    """Mean GT HR (bpm) over a time window [t_start, t_end]."""
    mask = (timestamps >= t_start) & (timestamps < t_end)
    if not mask.any():
        return float("nan")
    return float(np.mean(hr_frames[mask]))


# ── Feature extraction from all subjects ─────────────────────────────────────

def extract_all_features(archive_dir: str):
    """Walk all subjects, return (features, targets, subject_ids) arrays."""
    archive = Path(archive_dir)
    subjects = sorted(d for d in archive.iterdir() if d.is_dir())

    all_features = []
    all_targets  = []   # HR in Hz (not bpm) for regression
    all_subj_ids = []   # integer index per subject (for LOSO splits)

    print(f"Extracting features from {len(subjects)} subjects…")

    for sid, subj in enumerate(subjects):
        vid_path = subj / "vid.avi"
        gt_path  = subj / "ground_truth.txt"
        if not vid_path.exists() or not gt_path.exists():
            continue

        print(f"  [{sid+1:2d}/{len(subjects)}] {subj.name}", end="", flush=True)

        rgb, fps = extract_rgb_from_video(str(vid_path))
        if rgb is None:
            print(" — SKIP (video read failed)")
            continue

        hr_frames, timestamps = load_ground_truth(str(gt_path))
        total_sec = len(rgb) / fps

        win_frames    = int(fps * WIN_SEC)
        stride_frames = int(fps * STRIDE_SEC)
        min_frames    = int(fps * MIN_WIN_SEC)

        n_windows = 0
        start = 0
        while start + min_frames <= len(rgb):
            end = min(start + win_frames, len(rgb))
            seg = rgb[start:end]
            if len(seg) < min_frames:
                break

            t_start = start / fps
            t_end   = end   / fps
            gt_hr_bpm = gt_hr_for_window(hr_frames, timestamps, t_start, t_end)
            if np.isnan(gt_hr_bpm) or gt_hr_bpm < 45 or gt_hr_bpm > 180:
                start += stride_frames
                continue

            pulse = chrom_signal(seg, fps)
            feat  = psd_features(pulse, fps)

            all_features.append(feat)
            all_targets.append(gt_hr_bpm / 60.0)  # bpm → Hz
            all_subj_ids.append(sid)
            n_windows += 1
            start += stride_frames

        print(f" — {n_windows} windows")

    print(f"\nTotal: {len(all_features)} training samples from {len(set(all_subj_ids))} subjects\n")
    return (
        np.array(all_features,  dtype=np.float32),
        np.array(all_targets,   dtype=np.float32),
        np.array(all_subj_ids,  dtype=np.int32),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class HRNet(nn.Module):
    """Small MLP: normalised PSD (64-dim) → HR in Hz."""

    def __init__(self, n_bins: int = N_BINS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bins, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(features: np.ndarray, targets: np.ndarray, subj_ids: np.ndarray,
          epochs: int = 300, val_ratio: float = 0.2, seed: int = 42):
    """Train HRNet, return the best model (lowest val MAE)."""
    rng = np.random.default_rng(seed)

    # Subject-stratified train/val split (keep full subjects together)
    unique_subjs = np.unique(subj_ids)
    n_val_subjs  = max(1, int(len(unique_subjs) * val_ratio))
    val_subjs    = set(rng.choice(unique_subjs, n_val_subjs, replace=False))

    train_mask = np.array([sid not in val_subjs for sid in subj_ids])
    val_mask   = ~train_mask

    X_train = torch.tensor(features[train_mask])
    y_train = torch.tensor(targets[train_mask])
    X_val   = torch.tensor(features[val_mask])
    y_val   = torch.tensor(targets[val_mask])

    print(f"Train: {train_mask.sum()} samples | Val: {val_mask.sum()} samples")
    print(f"Val subjects: {sorted(val_subjs)}\n")

    train_ds = TensorDataset(X_train, y_train)
    loader   = DataLoader(train_ds, batch_size=32, shuffle=True)

    model     = HRNet(N_BINS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.HuberLoss(delta=0.05)   # 0.05 Hz ≈ 3 bpm — robust to outliers

    best_val_mae = float("inf")
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred  = model(X_val)
                val_mae   = float((val_pred - y_val).abs().mean()) * 60.0  # → bpm
                train_pred = model(X_train)
                train_mae  = float((train_pred - y_train).abs().mean()) * 60.0
            print(f"  Epoch {epoch:4d} | train MAE {train_mae:.2f} bpm | val MAE {val_mae:.2f} bpm")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\nBest val MAE: {best_val_mae:.2f} bpm")
    return model


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_bins":      N_BINS,
        "freq_low":    FREQ_LOW,
        "freq_high":   FREQ_HIGH,
        "freq_grid":   FREQ_GRID.tolist(),
    }, path)
    print(f"Model saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train rPPG spectral MLP on UBFC-rPPG")
    parser.add_argument("--archive", default="e:/ai-intern/archive")
    parser.add_argument("--epochs",  type=int, default=300)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    features, targets, subj_ids = extract_all_features(args.archive)
    if len(features) == 0:
        print("No training data extracted. Check archive path.")
        sys.exit(1)

    model = train(features, targets, subj_ids, epochs=args.epochs, seed=args.seed)
    save_model(model, MODEL_PATH)
    print("\nDone. Run benchmark_rppg.py to evaluate the trained model.")
