"""Cheating detector using personalised calibration data.

Key design principles
---------------------
* Every raw iris coordinate is passed through the affine transform BEFORE
  zone classification — no hardcoded pixel offsets.
* Signal 1 (horizontal scan): uses mean absolute x-DISPLACEMENT per frame
  (same units as baseline_variance = mean displacement), not position variance.
  Previous version compared position variance to displacement — dimensional mismatch.
* Signal 3 (periodic scan): Hann window applied before FFT to suppress spectral
  leakage.  Welch power-spectral-density then computed for more stable peak detection.
* neurodiversity_adjustment applied in both the stateful CheatingDetector class
  AND the stateless batch detect_cheating() function.
* calibration_quality_score < 0.6 exposes `needs_recalibration = True`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from services.video_analysis.calibration.calibration_runner import (
    apply_transform,
    load_calibration,
)
from services.video_analysis.gaze.zone_classifier import GazeZone, ZoneClassifier


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class CheatingFlags:
    horizontal_scan_detected: bool = False
    sustained_red_zone: bool       = False
    rapid_gaze_shift: bool         = False
    calibration_quality_low: bool  = False
    zone_sequence: List[str]       = field(default_factory=list)
    risk_level: str                = "low"   # "low" | "medium" | "high"


# ── Detector ─────────────────────────────────────────────────────────────────

class CheatingDetector:
    """
    Stateful detector for one candidate session.

    Parameters
    ----------
    session_id        : Must have a corresponding calibration JSON on disk.
    window_size       : Sliding window of recent frames (default 90 ≈ 3 s at 30 fps).
    red_zone_patience : Consecutive red-zone frames before flag is raised.
    """

    def __init__(
        self,
        session_id: str,
        window_size: int = 90,
        red_zone_patience: int = 45,
    ) -> None:
        cal = load_calibration(session_id)
        self.transform_matrix: list            = cal["transform_matrix"]
        self.baseline_variance: float          = cal["baseline_gaze_variance"]
        self.neurodiversity_adjustment: float  = cal["neurodiversity_adjustment"]
        self.calibration_quality_score: float  = cal["calibration_quality_score"]

        # Personalised threshold: 2× mean displacement = clear scan (not tiny tremor)
        self.h_scan_threshold: float = self.baseline_variance * 2.0

        self._classifier   = ZoneClassifier(session_id)
        self._zone_window: Deque[GazeZone]            = deque(maxlen=window_size)
        self._iris_window: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self._red_streak   = 0
        self._red_patience = red_zone_patience

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def needs_recalibration(self) -> bool:
        return self.calibration_quality_score < 0.6

    def process_frame(self, raw_iris: Tuple[float, float]) -> CheatingFlags:
        """Process one frame and return current cheat flags."""
        _ = apply_transform(raw_iris, self.transform_matrix)

        prev_iris: Optional[Tuple[float, float]] = (
            self._iris_window[-1] if self._iris_window else None
        )
        zone = self._classifier.classify(raw_iris, prev_iris)

        self._zone_window.append(zone)
        self._iris_window.append(raw_iris)

        if zone == GazeZone.RED:
            self._red_streak += 1
        else:
            self._red_streak = 0

        return self._evaluate()

    # ── Private ───────────────────────────────────────────────────────────────

    def _evaluate(self) -> CheatingFlags:
        flags = CheatingFlags(
            zone_sequence=[z.value for z in list(self._zone_window)[-10:]],
            calibration_quality_low=self.needs_recalibration,
        )

        flags.sustained_red_zone = self._red_streak >= self._red_patience

        # Horizontal scan: mean absolute x-displacement vs personalised threshold
        if len(self._iris_window) >= 10:
            recent = np.array(list(self._iris_window)[-30:])
            x_disp_mean = float(np.abs(np.diff(recent[:, 0])).mean())
            flags.horizontal_scan_detected = x_disp_mean > self.h_scan_threshold

        flags.rapid_gaze_shift = self._detect_rapid_shift()

        n_flags = sum([
            flags.horizontal_scan_detected,
            flags.sustained_red_zone,
            flags.rapid_gaze_shift,
        ])
        effective = n_flags / self.neurodiversity_adjustment
        if effective >= 2.0:
            flags.risk_level = "high"
        elif effective >= 1.0:
            flags.risk_level = "medium"

        return flags

    def _detect_rapid_shift(self) -> bool:
        if len(self._iris_window) < 2:
            return False
        iris_list = list(self._iris_window)
        last = np.array(iris_list[-1])
        prev = np.array(iris_list[-2])
        disp = float(np.linalg.norm(last - prev))
        return disp > 3.0 * self.baseline_variance


# ── Standalone batch detector (called from API per question response) ─────────

def detect_cheating(
    gaze_points: List[Tuple[float, float]],
    baseline_variance: float = 0.004,
    neurodiversity_adjustment: float = 1.0,
) -> dict:
    """
    Stateless batch cheating detector for one question's gaze samples.

    Runs 9 independent signal detectors and aggregates a risk score.
    neurodiversity_adjustment divides the final score to avoid penalising
    candidates with naturally higher gaze variance (consistent with the
    stateful CheatingDetector class).

    Signal weights (max total = 13):
    1.  Horizontal scan (x-displacement)  weight=1
    2.  Rapid gaze jumps                  weight=1
    3.  Periodic scan — Hann+Welch FFT    weight=2  (strong signal)
    4.  Directional sweeps (L→R→L)        weight=1
    5.  Gaze freeze                       weight=2  (very suspicious)
    6.  Extreme lateral gaze              weight=1
    7.  Robotic velocity consistency      weight=2  (strong signal)
    8.  Linear reading trajectory         weight=2  (strong signal)
    9.  Sustained downward gaze           weight=1
    """
    if len(gaze_points) < 5:
        return {"risk_level": "low", "cheat_score": 0, "reason": "insufficient_data"}

    pts = np.array(gaze_points, dtype=float)   # shape (N, 2)
    N   = len(pts)
    flags: dict = {}
    score = 0

    # ── 1. Horizontal scan ─────────────────────────────────────────────────────
    # Compare mean absolute x-DISPLACEMENT (same unit as baseline_variance)
    # NOT position variance (different unit — caused false positives).
    if N >= 2:
        x_disp_mean = float(np.abs(np.diff(pts[:, 0])).mean())
        h_scan = x_disp_mean > baseline_variance * 2.0
        flags["horizontal_scan"]   = h_scan
        flags["x_disp_mean"]       = round(x_disp_mean, 5)
        if h_scan:
            score += 1
    else:
        flags["horizontal_scan"] = False

    # ── 2. Rapid gaze jumps ────────────────────────────────────────────────────
    if N >= 2:
        displacements = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        rapid_shift = bool(np.any(displacements > 3.0 * baseline_variance))
        flags["rapid_gaze_shift"] = rapid_shift
        if rapid_shift:
            score += 1

    # ── 3. Periodic scan — Hann-windowed Welch PSD ────────────────────────────
    # Hann window eliminates spectral leakage that caused false "reading" flags
    # with the previous bare rfft.  Welch PSD averages over overlapping windows
    # for a more stable spectral estimate.
    # Gaze samples arrive at ~12.5 fps (80 ms frontend timeslice).
    if N >= 16:
        x_c = pts[:, 0] - pts[:, 0].mean()
        try:
            from scipy.signal import welch
            # nperseg = min(N, 32) gives ~5 freq bins in the cardiac band
            nperseg = min(N, max(16, N // 2))
            freqs_w, psd = welch(x_c, fs=12.5, nperseg=nperseg, window="hann")
            band = (freqs_w >= 0.3) & (freqs_w <= 3.5)
            if band.any():
                peak_power  = float(psd[band].max())
                total_power = float(psd.sum()) + 1e-9
                periodic_ratio = peak_power / total_power
                flags["periodic_scan"]  = periodic_ratio > 0.40
                flags["periodic_ratio"] = round(periodic_ratio, 3)
                if flags["periodic_scan"]:
                    score += 2
        except Exception:
            # scipy unavailable — fall back to Hann-windowed rfft
            window  = np.hanning(N)
            x_win   = x_c * window
            fft_mag = np.abs(np.fft.rfft(x_win))
            fft_mag[0] = 0
            freqs = np.fft.rfftfreq(N, d=0.08)
            band  = (freqs >= 0.3) & (freqs <= 3.5)
            if band.any():
                peak_power  = float(fft_mag[band].max())
                total_power = float(fft_mag[1:].sum()) + 1e-9
                periodic_ratio = peak_power / total_power
                flags["periodic_scan"]  = periodic_ratio > 0.40
                flags["periodic_ratio"] = round(periodic_ratio, 3)
                if flags.get("periodic_scan", False):
                    score += 2

    # ── 4. Directional sweeps (L→R→L reversal count) ──────────────────────────
    if N >= 10:
        x_deltas    = np.diff(pts[:, 0])
        significant = x_deltas[np.abs(x_deltas) > 0.015]
        if len(significant) >= 4:
            sign_changes = int(np.sum(np.diff(np.sign(significant)) != 0))
            sweep_hz = sign_changes / max(N * 0.08, 1e-3)
            directional_sweep = 0.5 <= sweep_hz <= 5.0
            flags["directional_sweep"] = directional_sweep
            flags["sweep_rate_hz"]     = round(sweep_hz, 2)
            if directional_sweep:
                score += 1

    # ── 5. Gaze freeze (suspiciously low total displacement variance) ──────────
    # Uses mean displacement (same units as baseline_variance) rather than
    # position variance to avoid the coordinate-space mismatch.
    if N >= 10:
        all_disps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        mean_disp = float(all_disps.mean())
        gaze_frozen = mean_disp < baseline_variance * 0.05
        flags["gaze_freeze"] = gaze_frozen
        flags["mean_displacement"] = round(mean_disp, 5)
        if gaze_frozen:
            score += 2

    # ── 6. Extreme lateral gaze ────────────────────────────────────────────────
    if N >= 5:
        ext_left  = float(np.mean(pts[:, 0] < 0.12))
        ext_right = float(np.mean(pts[:, 0] > 0.88))
        extreme_lateral = (ext_left + ext_right) > 0.15
        flags["extreme_lateral_gaze"] = extreme_lateral
        flags["extreme_left_pct"]     = round(ext_left,  3)
        flags["extreme_right_pct"]    = round(ext_right, 3)
        if extreme_lateral:
            score += 1

    # ── 7. Robotic velocity consistency ───────────────────────────────────────
    # Natural gaze: mix of fast saccades + slow fixations → high velocity CV.
    # Scripted/robotic: unnaturally uniform speed → low CV.
    if N >= 10:
        velocities = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        mean_vel   = float(velocities.mean()) + 1e-9
        vel_cv     = float(velocities.std()) / mean_vel
        robotic    = vel_cv < 0.25 and mean_vel > baseline_variance * 0.5
        flags["robotic_velocity"] = robotic
        flags["velocity_cv"]      = round(vel_cv, 3)
        if robotic:
            score += 2

    # ── 8. Linear reading trajectory ──────────────────────────────────────────
    if N >= 12:
        try:
            t        = np.arange(N, dtype=float)
            coeffs   = np.polyfit(t, pts[:, 0], 1)
            pred_x   = np.polyval(coeffs, t)
            res_var  = float(np.var(pts[:, 0] - pred_x))
            x_var_tot = float(np.var(pts[:, 0])) + 1e-9
            linearity = 1.0 - res_var / x_var_tot
            y_var     = float(np.var(pts[:, 1]))
            # y_var threshold: displacement-based to match coordinate space
            linear_read = linearity > 0.75 and y_var < (baseline_variance ** 2) * 5.0
            flags["linear_reading_trajectory"] = linear_read
            flags["trajectory_linearity"]      = round(linearity, 3)
            if linear_read:
                score += 2
        except Exception:
            pass

    # ── 9. Sustained downward gaze ─────────────────────────────────────────────
    if N >= 5:
        down_frac = float(np.mean(pts[:, 1] > 0.80))
        flags["sustained_downward_gaze"] = down_frac > 0.20
        flags["downward_gaze_pct"]       = round(down_frac, 3)
        if down_frac > 0.20:
            score += 1

    # ── Risk level (neurodiversity-adjusted) ───────────────────────────────────
    # Divide by neurodiversity_adjustment so candidates with naturally higher
    # gaze variance are not unfairly penalised (same logic as stateful class).
    effective_score = score / max(neurodiversity_adjustment, 1.0)
    if effective_score >= 5:
        risk_level = "high"
    elif effective_score >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    flags["risk_level"]       = risk_level
    flags["cheat_score"]      = score
    flags["effective_score"]  = round(effective_score, 2)
    return flags


# ── Head pose cheating detector ───────────────────────────────────────────────

def analyze_head_pose(
    head_pose_samples: List[Dict[str, float]],
    neurodiversity_adjustment: float = 1.0,
    face_absent_pct: float = 0.0,
) -> dict:
    """
    Analyse per-frame head pose (yaw / pitch) captured by MediaPipe FaceMesh.

    Works for both sources:
    - Browser real-time: geometric ratio (nose offset / half face width)
    - Post-processing:   solvePnP Euler angles normalised by 45 ° → same scale

    ``yaw``   – normalised horizontal rotation. 0 = straight, ±1 ≈ ±45 °.
    ``pitch`` – normalised vertical rotation.   0 = straight, positive = down.
    ``face_absent_pct`` – fraction of total tracking frames where NO face was
                          detected (candidate off-screen or fully turned away).

    Signals (max raw score = 7):
    A.  Sustained head turn   (left + right > 25 % of frames)     weight=2
    B.  Predominantly off-axis (|yaw| > 0.40 for > 15 % frames)   weight=2
    C.  Erratic head movement  (yaw std-dev > 0.15)                weight=1
    D.  Face absent            (no face in > 20 % of frames)       weight=2  ← real
    """
    flags: dict = {
        "head_pose_available":  len(head_pose_samples) >= 5,
        "face_absent_pct":      round(float(face_absent_pct), 4),
        "face_absent_flagged":  face_absent_pct > 0.20,
    }

    score = 0

    # ── D. Face absent signal (runs even when pose samples are sparse) ────────
    if face_absent_pct > 0.20:
        score += 2

    if len(head_pose_samples) < 5:
        flags.update({
            "sustained_head_turn":  False,
            "head_turn_left_pct":   0.0,
            "head_turn_right_pct":  0.0,
            "erratic_head_movement": False,
            "forward_facing_pct":   1.0,
            "head_pose_score":      score,
            "head_pose_effective_score": round(score / max(neurodiversity_adjustment, 1.0), 2),
        })
        return flags

    yaws = np.array([float(s.get("yaw", 0.0)) for s in head_pose_samples])
    N    = len(yaws)

    # ── A. Sustained head turn ────────────────────────────────────────────────
    # 0.25 normalised ≈ ~11° geometric or ~11° solvePnP after /45 normalisation.
    # A candidate reading notes typically turns 15–30 °.
    YAW_TURN  = 0.25
    left_pct  = float(np.mean(yaws >  YAW_TURN))
    right_pct = float(np.mean(yaws < -YAW_TURN))
    sustained = (left_pct + right_pct) > 0.25
    flags["head_turn_left_pct"]  = round(left_pct,  3)
    flags["head_turn_right_pct"] = round(right_pct, 3)
    flags["sustained_head_turn"] = sustained
    if sustained:
        score += 2

    # ── B. Predominantly off-axis (|yaw| > 0.40 ≈ 18 °) ─────────────────────
    YAW_EXTREME        = 0.40
    extreme_pct        = float(np.mean(np.abs(yaws) > YAW_EXTREME))
    predominantly_off  = extreme_pct > 0.15
    flags["extreme_head_turn_pct"]  = round(extreme_pct, 3)
    flags["predominantly_off_axis"] = predominantly_off
    if predominantly_off:
        score += 2

    # ── C. Erratic head movement ──────────────────────────────────────────────
    if N >= 10:
        yaw_std = float(np.std(yaws))
        erratic = yaw_std > 0.15
        flags["erratic_head_movement"] = erratic
        flags["head_yaw_std"]          = round(yaw_std, 3)
        if erratic:
            score += 1

    flags["forward_facing_pct"] = round(float(np.mean(np.abs(yaws) < YAW_TURN)), 3)

    effective = score / max(neurodiversity_adjustment, 1.0)
    flags["head_pose_score"]           = score
    flags["head_pose_effective_score"] = round(effective, 2)
    return flags
