"""CHROM rPPG algorithm for non-contact heart rate and HRV estimation.

Reference: De Haan & Jeanne (2013) — Robust Pulse Rate From Chrominance-Based rPPG.

Signal processing improvements
-------------------------------
- Butterworth bandpass filter (0.75–3 Hz, order 4) replaces brick-wall FFT
  zeroing.  Brick-wall zeroing causes ringing artefacts in the time domain that
  corrupt the pulse signal; Butterworth is smooth and stable.
- Motion-frame rejection: frames where the frame-to-frame image delta exceeds a
  threshold are excluded before CHROM processing, preventing head-movement
  artefacts from contaminating the cardiac signal.
- Face ROI uses OpenCV Haar cascade → forehead region (top 35 % of face) for
  a stronger rPPG signal with less clothing/background contamination.
- Sliding-window CHROM (10 s windows, median HR across windows) for stability.
- RMSSD returned as None when fewer than 3 R-peaks are found — not 42.0.
- data_available=False returned clearly so downstream code never treats
  placeholder values as real measurements.
- Trained spectral MLP (rppg_model.pt) replaces FFT peak-picking when
  available, fixing the sub-harmonic locking problem that causes 30–60 bpm
  underestimates on some subjects.  Falls back to CHROM FFT if model absent.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_DEFAULTS: Dict = {
    "avg_hrv_rmssd": None,
    "hr_bpm": None,
    "stress_spike_detected": False,
    "data_available": False,
}

# Butterworth bandpass limits (cardiac band)
_BP_LOW_HZ  = 0.75
_BP_HIGH_HZ = 3.0

# Spectral MLP config (must match train_rppg_model.py)
_N_BINS   = 64
_FREQ_GRID = np.linspace(_BP_LOW_HZ, _BP_HIGH_HZ, _N_BINS)
_MODEL_PATH = Path(__file__).parent / "rppg_model.pt"


# ── Face cascade (lazy, cached) ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_face_cascade():
    try:
        import cv2
        cc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return cc if not cc.empty() else None
    except Exception:
        return None


# ── Butterworth bandpass (cached per fps) ─────────────────────────────────────

@lru_cache(maxsize=8)
def _butter_bandpass(fps: float, order: int = 4):
    """Return (b, a) Butterworth bandpass coefficients for the cardiac band."""
    try:
        from scipy.signal import butter
        nyq = fps / 2.0
        low  = _BP_LOW_HZ  / nyq
        high = _BP_HIGH_HZ / nyq
        low  = max(1e-3, min(low,  0.999))
        high = max(1e-3, min(high, 0.999))
        if low >= high:
            return None
        return butter(order, [low, high], btype="band")
    except Exception:
        return None


def _apply_bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    """Apply Butterworth bandpass; fall back to FFT-zeroing if scipy unavailable."""
    coeffs = _butter_bandpass(fps)
    if coeffs is not None:
        try:
            from scipy.signal import filtfilt
            b, a = coeffs
            return filtfilt(b, a, signal)
        except Exception:
            pass

    # Fallback: FFT-domain zeroing (original approach)
    N = len(signal)
    freqs = np.fft.fftfreq(N, d=1.0 / fps)
    F = np.fft.fft(signal)
    mask = (np.abs(freqs) < _BP_LOW_HZ) | (np.abs(freqs) > _BP_HIGH_HZ)
    F[mask] = 0
    return np.real(np.fft.ifft(F))


# ── Trained spectral MLP (HR estimator) ──────────────────────────────────────

@lru_cache(maxsize=1)
def _load_hr_model():
    """Load HRNet from rppg_model.pt if it exists; return None otherwise."""
    if not _MODEL_PATH.exists():
        return None
    try:
        import torch
        import torch.nn as nn

        ckpt = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=False)
        n_bins = ckpt.get("n_bins", _N_BINS)

        class _HRNet(nn.Module):
            def __init__(self, n: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 1),
                )
            def forward(self, x):
                return self.net(x).squeeze(-1)

        model = _HRNet(n_bins)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[Examiney][rPPG] Loaded trained HRNet from {_MODEL_PATH.name}")
        return model
    except Exception as exc:
        print(f"[Examiney][rPPG] Could not load HRNet: {exc} — falling back to CHROM FFT")
        return None


def _psd_features(pulse: np.ndarray, fps: float) -> np.ndarray:
    """Normalised PSD interpolated onto _FREQ_GRID (fps-agnostic, 64-dim)."""
    N     = len(pulse)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    psd   = np.abs(np.fft.rfft(pulse)) ** 2
    feat  = np.interp(_FREQ_GRID, freqs, psd, left=0.0, right=0.0)
    total = feat.sum()
    if total > 1e-12:
        feat /= total
    return feat.astype(np.float32)


def _model_hr_freq(pulse: np.ndarray, fps: float) -> Optional[float]:
    """Run trained HRNet on pulse signal; return HR in Hz, or None if unavailable."""
    model = _load_hr_model()
    if model is None:
        return None
    try:
        import torch
        feat   = torch.tensor(_psd_features(pulse, fps)).unsqueeze(0)  # (1, 64)
        with torch.no_grad():
            hr_hz = float(model(feat).item())
        # Clamp to physiological cardiac band
        hr_hz = max(_BP_LOW_HZ, min(hr_hz, _BP_HIGH_HZ))
        return hr_hz
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_rppg_from_video(video_path: str) -> Dict:
    """Run CHROM rPPG on *video_path*.

    Returns:
        {
            "avg_hrv_rmssd": float | None,
            "hr_bpm": float | None,
            "stress_spike_detected": bool,
            "data_available": bool,
        }
    """
    try:
        import cv2
    except ImportError:
        print("[Examiney][rPPG] cv2 not installed — returning defaults.")
        return _DEFAULTS.copy()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _DEFAULTS.copy()

    # Browser WebM recordings (MediaRecorder) often report fps=1000 via OpenCV.
    # Clamp to a physiologically sane range so min_frames stays reasonable.
    _raw_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = max(1.0, min(_raw_fps, 60.0))
    rgb_signals: List[np.ndarray] = []
    prev_gray: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Motion-frame rejection ──────────────────────────────────────────
        # Exclude frames where the candidate moves sharply (head turn, gesture)
        # as these contaminate the rPPG signal with motion artefacts.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # Mean absolute difference between consecutive frames
            frame_delta = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean())
            if frame_delta > 12.0:     # empirical threshold (~5 % of 255)
                prev_gray = gray
                continue               # skip high-motion frame
        prev_gray = gray

        rgb = _mean_face_rgb(frame)
        if rgb is not None:
            rgb_signals.append(rgb)

    cap.release()

    # Minimum: 2.5 s of usable face signal — enough for ≥2 cardiac cycles at 60 bpm.
    # Short interview clips are often 3–6 s; the old threshold of 5 s rejected them all.
    min_frames = max(50, int(fps * 2.5))
    if len(rgb_signals) < min_frames:
        print(
            f"[Examiney][rPPG] Only {len(rgb_signals)} usable frames "
            f"(need {min_frames}) — insufficient data."
        )
        return _DEFAULTS.copy()

    try:
        rmssd, hr_bpm, stress = _chrom_windowed(
            np.array(rgb_signals, dtype=np.float64), fps
        )
        return {
            "avg_hrv_rmssd":       rmssd,
            "hr_bpm":              hr_bpm,
            "stress_spike_detected": stress,
            "data_available":      True,
        }
    except Exception as exc:
        print(f"[Examiney][rPPG] CHROM failed: {exc}")
        return _DEFAULTS.copy()


# ── Private helpers ───────────────────────────────────────────────────────────

def _mean_face_rgb(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return mean [R, G, B] of the forehead region.

    Uses Haar cascade to locate the face; extracts the forehead (top 35 % of
    bounding box) which has the strongest rPPG signal and least contamination.
    Falls back to a fixed central ROI when face detection fails.
    """
    import cv2

    h, w = frame.shape[:2]
    cascade = _get_face_cascade()

    if cascade is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3,
            minSize=(int(w * 0.1), int(h * 0.1)),
        )
        if len(faces) > 0:
            # Pick largest face
            x, y, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            forehead_h = max(1, int(fh * 0.35))
            roi = frame[y: y + forehead_h, x: x + fw]
            if roi.size > 0:
                bgr = roi.reshape(-1, 3).mean(axis=0)
                return bgr[::-1].copy()  # BGR → RGB

    # Fixed central fallback
    roi = frame[int(h * 0.20): int(h * 0.55), int(w * 0.25): int(w * 0.75)]
    if roi.size == 0:
        return None
    bgr = roi.reshape(-1, 3).mean(axis=0)
    return bgr[::-1].copy()


def _chrom_single(
    rgb: np.ndarray, fps: float
) -> Tuple[Optional[float], np.ndarray]:
    """CHROM decomposition on a single segment.

    Returns (dominant_hr_hz | None, filtered_pulse_signal).

    HR estimation strategy:
      1. Try trained HRNet (spectral MLP) — avoids sub-harmonic locking.
      2. Fall back to FFT peak-picking if model unavailable.
    """
    means = rgb.mean(axis=0)
    means = np.where(means == 0, 1e-6, means)
    norm  = rgb / means

    R, G, B = norm[:, 0], norm[:, 1], norm[:, 2]
    Xs = 3 * R - 2 * G
    Ys = 1.5 * R + G - 1.5 * B
    alpha = Xs.std() / (Ys.std() or 1e-6)
    S = Xs - alpha * Ys

    # Butterworth bandpass filter
    sig_filt = _apply_bandpass(S, fps)

    # ── HR estimation: trained model (preferred) ───────────────────────────
    hr_freq = _model_hr_freq(sig_filt, fps)
    if hr_freq is not None:
        return hr_freq, sig_filt

    # ── HR estimation: FFT peak-picking (fallback) ─────────────────────────
    N     = len(sig_filt)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    F_mag = np.abs(np.fft.rfft(sig_filt))
    pos   = (freqs >= _BP_LOW_HZ) & (freqs <= _BP_HIGH_HZ)
    if not pos.any() or F_mag[pos].max() < 1e-9:
        return None, sig_filt

    hr_freq = float(freqs[pos][F_mag[pos].argmax()])
    return hr_freq, sig_filt


def _chrom_windowed(
    rgb: np.ndarray,
    fps: float,
    window_sec: float = 10.0,
) -> Tuple[Optional[float], Optional[float], bool]:
    """Sliding-window CHROM → (rmssd_ms | None, hr_bpm | None, stress_bool).

    10-second non-overlapping windows; median HR across valid windows.
    RMSSD computed from R-peaks on the full concatenated filtered signal.
    """
    # For short videos use the whole signal as one window rather than 10-s chunks.
    total_frames = len(rgb)
    effective_window_sec = min(window_sec, total_frames / fps * 0.95)
    win_frames = max(1, int(fps * effective_window_sec))
    n_windows  = max(1, total_frames // win_frames)

    # Minimum segment: 2.5 s (same as outer gate) — enough for cardiac frequency detection.
    min_seg_frames = max(50, int(fps * 2.5))

    hr_freqs: List[float] = []
    all_pulse: List[float] = []

    for i in range(n_windows):
        seg = rgb[i * win_frames: (i + 1) * win_frames]
        if len(seg) < min_seg_frames:
            continue
        try:
            hr_f, pulse = _chrom_single(seg, fps)
            if hr_f is not None:
                hr_freqs.append(hr_f)
            all_pulse.extend(pulse.tolist())
        except Exception:
            continue

    if not hr_freqs:
        raise ValueError("No valid CHROM windows produced a cardiac frequency")

    hr_bpm: Optional[float] = round(float(np.median(hr_freqs)) * 60.0, 2)

    # RMSSD from R-peaks on full concatenated pulse
    rmssd: Optional[float] = None
    if all_pulse:
        try:
            from scipy.signal import find_peaks
            pulse_arr = np.array(all_pulse)
            # prominence filter removes shallow noise peaks
            peaks, _ = find_peaks(
                pulse_arr,
                distance=int(fps * 0.4),
                prominence=pulse_arr.std() * 0.3,
            )
            if len(peaks) >= 3:
                rr_ms  = np.diff(peaks) / fps * 1000.0
                # Physiological RR-interval sanity gate (300–2000 ms = 30–200 bpm)
                rr_ms  = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
                if len(rr_ms) >= 2:
                    rmssd = round(float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))), 2)
        except Exception:
            pass  # scipy unavailable or peak detection failed

    stress = bool(
        (rmssd is not None and rmssd < 20.0)
        or (hr_bpm is not None and hr_bpm > 100.0)
    )
    return rmssd, hr_bpm, stress
