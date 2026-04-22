"""Head pose extraction from recorded interview videos — MediaPipe Tasks API.

MediaPipe 0.10+ removed ``mp.solutions``; this module uses the Tasks API
(``mediapipe.tasks.python.vision.FaceLandmarker``) with
``output_facial_transformation_matrixes=True``, which provides a 4×4 rigid
transformation matrix directly — no solvePnP needed.

Euler angles are extracted from that matrix and normalised by 45° so the
resulting yaw/pitch values are on the same [-1, 1] scale as the browser-side
geometric-ratio computation, keeping ``analyze_head_pose()`` thresholds
consistent across both paths.

Face-absent frames (no face detected) are tracked separately and returned as
``face_absent_pct`` — the primary signal for "candidate left the screen".
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List

import numpy as np

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "face_landmarker.task"
)
_MODEL_PATH = os.path.abspath(_MODEL_PATH)

_NORM_DIVISOR = 45.0   # real degrees → normalised unit


def _rotation_matrix_to_euler(R: np.ndarray):
    """ZYX Euler angles in degrees from a 3×3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2( R[2, 1],  R[2, 2])
        yaw   = math.atan2(-R[2, 0],  sy)
        roll  = math.atan2( R[1, 0],  R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2],  R[1, 1])
        yaw   = math.atan2(-R[2, 0],  sy)
        roll  = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def extract_head_pose_from_video(
    video_path: str,
    every_n: int = 3,
) -> Dict[str, Any]:
    """Extract per-frame head pose from *video_path* using MediaPipe Tasks API.

    Returns::

        {
            "samples":         List[{"yaw": float, "pitch": float,
                                     "yaw_deg": float, "pitch_deg": float}],
            "face_absent_pct": float,
            "total_frames":    int,
        }
    """
    _empty = {"samples": [], "face_absent_pct": 0.0, "total_frames": 0}

    try:
        import cv2
    except ImportError:
        print("[Examiney][HeadPose] OpenCV not installed.")
        return _empty

    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("[Examiney][HeadPose] mediapipe.tasks not available.")
        return _empty

    if not os.path.exists(_MODEL_PATH):
        print(f"[Examiney][HeadPose] model not found: {_MODEL_PATH}")
        return _empty

    from services.video_analysis.gaze.gazefollower_runner import _extract_frames_cv2
    frames: List[Any] = _extract_frames_cv2(video_path, every_n=every_n)

    if not frames:
        print(f"[Examiney][HeadPose] 0 frames decoded from {video_path}")
        return _empty

    total_frames = len(frames)
    face_absent  = 0
    samples: List[Dict] = []

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        for frame in frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp_vision.Image(
                    image_format=mp_vision.ImageFormat.SRGB, data=rgb
                )
                result = landmarker.detect(mp_img)

                if not result.face_landmarks:
                    face_absent += 1
                    continue

                # Facial transformation matrix (4×4 world→camera rigid transform)
                if not result.facial_transformation_matrixes:
                    face_absent += 1
                    continue

                mat = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
                R   = mat[:3, :3]
                pitch_deg, yaw_deg, _ = _rotation_matrix_to_euler(R)

                yaw_n   = float(np.clip(yaw_deg   / _NORM_DIVISOR, -2.5, 2.5))
                pitch_n = float(np.clip(pitch_deg / _NORM_DIVISOR, -2.5, 2.5))

                samples.append({
                    "yaw":       round(yaw_n,     4),
                    "pitch":     round(pitch_n,   4),
                    "yaw_deg":   round(yaw_deg,   1),
                    "pitch_deg": round(pitch_deg, 1),
                })

            except Exception as e:
                print(f"[Examiney][HeadPose] frame error: {e}")
                face_absent += 1

    face_absent_pct = round(face_absent / max(total_frames, 1), 4)
    print(
        f"[Examiney][HeadPose] {len(samples)} pose samples | "
        f"face absent {face_absent}/{total_frames} ({face_absent_pct:.1%})"
    )
    return {
        "samples":         samples,
        "face_absent_pct": face_absent_pct,
        "total_frames":    total_frames,
    }
