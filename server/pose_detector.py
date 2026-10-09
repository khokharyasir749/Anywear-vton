"""
Anywear VTO - Pose Detector Module
Integrates MediaPipe Pose (Tasks API & Solutions API compatible) with Adaptive Exponential
Moving Average (EMA) smoothing, body orientation calculation, and key landmark extraction
(shoulders, elbows, wrists, hips, neck, waist).
"""

import math
import time
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("anywear-vton.pose")

MODEL_TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_TASK_FILENAME = "pose_landmarker_lite.task"


@dataclass
class PoseData:
    """Encapsulates smoothed 2D/3D body landmarks and calculated geometric properties."""
    detected: bool = False
    confidence: float = 0.0
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Key Anchor Points (in image pixel space)
    neck: Tuple[float, float] = (0.0, 0.0)
    chest: Tuple[float, float] = (0.0, 0.0)
    waist: Tuple[float, float] = (0.0, 0.0)
    left_shoulder: Tuple[float, float] = (0.0, 0.0)
    right_shoulder: Tuple[float, float] = (0.0, 0.0)
    left_elbow: Tuple[float, float] = (0.0, 0.0)
    right_elbow: Tuple[float, float] = (0.0, 0.0)
    left_wrist: Tuple[float, float] = (0.0, 0.0)
    right_wrist: Tuple[float, float] = (0.0, 0.0)
    left_hip: Tuple[float, float] = (0.0, 0.0)
    right_hip: Tuple[float, float] = (0.0, 0.0)
    left_knee: Tuple[float, float] = (0.0, 0.0)
    right_knee: Tuple[float, float] = (0.0, 0.0)
    left_ankle: Tuple[float, float] = (0.0, 0.0)
    right_ankle: Tuple[float, float] = (0.0, 0.0)

    # Derived Biometric Dimensions
    shoulder_width: float = 0.0
    torso_height: float = 0.0
    hip_width: float = 0.0
    tilt_angle_deg: float = 0.0
    scale_factor: float = 1.0
    yaw_ratio: float = 0.0  # Body facing orientation (left vs right vs frontal)
    segmentation_mask: Optional[np.ndarray] = None


class PoseDetector:
    """
    High-performance MediaPipe Pose detector with adaptive EMA smoothing.
    Optimized for real-time video feeds with zero-jitter landmark tracking.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
        base_ema_alpha: float = 0.55
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.base_ema_alpha = base_ema_alpha
        self.prev_landmarks: Dict[str, Tuple[float, float]] = {}

        self.landmarker = None
        self.api_type = None  # "tasks" or "solutions"

        self._init_mediapipe()

    def _init_mediapipe(self):
        """Initializes MediaPipe Tasks or fallback Solutions API."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Locate or download pose_landmarker_lite.task
            model_path = os.path.join(os.path.dirname(__file__), MODEL_TASK_FILENAME)
            if not os.path.exists(model_path):
                logger.info("Downloading %s model asset...", MODEL_TASK_FILENAME)
                urllib.request.urlretrieve(MODEL_TASK_URL, model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=True,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.mp = mp
            self.api_type = "tasks"
            logger.info("MediaPipe Tasks PoseLandmarker initialized successfully.")
            return
        except Exception as e:
            logger.info("MediaPipe Tasks API not available, attempting Solutions API fallback: %s", e)

        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self.mp_pose = mp.solutions.pose
                self.landmarker = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    smooth_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.api_type = "solutions"
                logger.info("MediaPipe Solutions Pose initialized successfully.")
                return
        except Exception as e2:
            logger.warning("Could not initialize MediaPipe Solutions Pose: %s", e2)

        logger.warning("Pose detector running in simulated/fallback mode.")

    def _apply_adaptive_ema(
        self, key: str, current_val: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Adaptive Exponential Moving Average:
        If displacement between frames is small (micro-jitter), uses smaller alpha for stability.
        If displacement is large (rapid motion), dynamically ramps up alpha to prevent lag.
        """
        if key not in self.prev_landmarks:
            self.prev_landmarks[key] = current_val
            return current_val

        prev_x, prev_y = self.prev_landmarks[key]
        curr_x, curr_y = current_val

        dist = math.hypot(curr_x - prev_x, curr_y - prev_y)

        # Adaptive alpha: between 0.35 (stationary stability) and 0.90 (high-speed response)
        alpha = np.clip(self.base_ema_alpha + (dist / 60.0) * 0.35, 0.35, 0.90)

        smoothed_x = alpha * curr_x + (1.0 - alpha) * prev_x
        smoothed_y = alpha * curr_y + (1.0 - alpha) * prev_y

        smoothed = (float(smoothed_x), float(smoothed_y))
        self.prev_landmarks[key] = smoothed
        return smoothed

    def reset_smoothing(self):
        """Clears smoothed cache (e.g. when user leaves frame)."""
        self.prev_landmarks.clear()

    def detect(self, frame: np.ndarray) -> PoseData:
        """
        Processes a BGR image frame and returns a smoothed PoseData object.
        """
        if frame is None or self.landmarker is None:
            return PoseData(detected=False)

        now = time.time()
        if hasattr(self, "_last_detect_time") and (now - self._last_detect_time < 0.035) and hasattr(self, "_last_pose_result"):
            return self._last_pose_result

        h, w = frame.shape[:2]

        try:
            landmarks_list = None

            seg_mask = None
            if self.api_type == "tasks":
                # MediaPipe Tasks API
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
                results = self.landmarker.detect(mp_image)
                if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
                    landmarks_list = results.pose_landmarks[0]
                    if getattr(results, "segmentation_masks", None) and len(results.segmentation_masks) > 0:
                        try:
                            seg_mask = results.segmentation_masks[0].numpy_view()
                        except Exception:
                            seg_mask = None

            elif self.api_type == "solutions":
                # Legacy Solutions API
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.landmarker.process(rgb)
                if results and results.pose_landmarks:
                    landmarks_list = results.pose_landmarks.landmark

            if not landmarks_list or len(landmarks_list) < 29:
                self.reset_smoothing()
                return PoseData(detected=False)

            # Landmarks:
            # 11: Left Shoulder, 12: Right Shoulder
            # 13: Left Elbow,    14: Right Elbow
            # 15: Left Wrist,    16: Right Wrist
            # 23: Left Hip,      24: Right Hip
            # 25: Left Knee,     26: Right Knee
            # 27: Left Ankle,    28: Right Ankle
            # 0:  Nose
            l_sh_raw = landmarks_list[11]
            r_sh_raw = landmarks_list[12]
            l_hip_raw = landmarks_list[23]
            r_hip_raw = landmarks_list[24]

            # Visibility check
            vis_sum = getattr(l_sh_raw, 'visibility', 1.0) + \
                      getattr(r_sh_raw, 'visibility', 1.0) + \
                      getattr(l_hip_raw, 'visibility', 1.0) + \
                      getattr(r_hip_raw, 'visibility', 1.0)
            avg_vis = vis_sum / 4.0

            if avg_vis < 0.40:
                return PoseData(detected=False, confidence=avg_vis)

            def to_px(lm):
                return (float(lm.x * w), float(lm.y * h))

            raw_map = {
                "left_shoulder": to_px(l_sh_raw),
                "right_shoulder": to_px(r_sh_raw),
                "left_elbow": to_px(landmarks_list[13]),
                "right_elbow": to_px(landmarks_list[14]),
                "left_wrist": to_px(landmarks_list[15]),
                "right_wrist": to_px(landmarks_list[16]),
                "left_hip": to_px(l_hip_raw),
                "right_hip": to_px(r_hip_raw),
                "left_knee": to_px(landmarks_list[25]),
                "right_knee": to_px(landmarks_list[26]),
                "left_ankle": to_px(landmarks_list[27]),
                "right_ankle": to_px(landmarks_list[28]),
                "nose": to_px(landmarks_list[0]),
            }

            # Apply adaptive EMA smoothing to eliminate micro-jitter
            smoothed = {}
            for name, pt in raw_map.items():
                smoothed[name] = self._apply_adaptive_ema(name, pt)

            l_sh = smoothed["left_shoulder"]
            r_sh = smoothed["right_shoulder"]
            l_hip = smoothed["left_hip"]
            r_hip = smoothed["right_hip"]

            # Neck = Midpoint between left & right shoulder
            neck = ((l_sh[0] + r_sh[0]) * 0.5, (l_sh[1] + r_sh[1]) * 0.5)
            # Waist = Midpoint between left & right hip
            waist = ((l_hip[0] + r_hip[0]) * 0.5, (l_hip[1] + r_hip[1]) * 0.5)
            # Chest = 30% from neck down towards waist
            chest = (
                neck[0] + 0.30 * (waist[0] - neck[0]),
                neck[1] + 0.30 * (waist[1] - neck[1])
            )

            smoothed["neck"] = neck
            smoothed["waist"] = waist
            smoothed["chest"] = chest

            # Derived Metrics
            shoulder_width = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
            torso_height = math.hypot(waist[0] - neck[0], waist[1] - neck[1])
            hip_width = math.hypot(l_hip[0] - r_hip[0], l_hip[1] - r_hip[1])

            dx = l_sh[0] - r_sh[0]
            dy = l_sh[1] - r_sh[1]
            tilt_angle_deg = math.degrees(math.atan2(dy, dx))
            scale_factor = shoulder_width / 160.0

            z_diff = getattr(l_sh_raw, 'z', 0.0) - getattr(r_sh_raw, 'z', 0.0)

            result_data = PoseData(
                detected=True,
                confidence=float(avg_vis),
                landmarks=smoothed,
                neck=neck,
                chest=chest,
                waist=waist,
                left_shoulder=l_sh,
                right_shoulder=r_sh,
                left_elbow=smoothed["left_elbow"],
                right_elbow=smoothed["right_elbow"],
                left_wrist=smoothed["left_wrist"],
                right_wrist=smoothed["right_wrist"],
                left_hip=l_hip,
                right_hip=r_hip,
                left_knee=smoothed["left_knee"],
                right_knee=smoothed["right_knee"],
                left_ankle=smoothed["left_ankle"],
                right_ankle=smoothed["right_ankle"],
                shoulder_width=shoulder_width,
                torso_height=torso_height,
                hip_width=hip_width,
                tilt_angle_deg=tilt_angle_deg,
                scale_factor=scale_factor,
                yaw_ratio=float(z_diff),
                segmentation_mask=seg_mask
            )
            self._last_detect_time = now
            self._last_pose_result = result_data
            return result_data

        except Exception as e:
            logger.error("Pose detection error: %s", e)
            return PoseData(detected=False)

    def draw_skeleton(self, frame: np.ndarray, pose: PoseData) -> np.ndarray:
        """Visualizes detected skeleton and anchor points with futuristic cyber styling."""
        if not pose.detected:
            return frame

        overlay = frame.copy()

        limbs = [
            (pose.left_shoulder, pose.right_shoulder),
            (pose.left_shoulder, pose.left_elbow),
            (pose.left_elbow, pose.left_wrist),
            (pose.right_shoulder, pose.right_elbow),
            (pose.right_elbow, pose.right_wrist),
            (pose.left_shoulder, pose.left_hip),
            (pose.right_shoulder, pose.right_hip),
            (pose.left_hip, pose.right_hip),
            (pose.left_hip, pose.left_knee),
            (pose.right_hip, pose.right_knee),
        ]

        for p1, p2 in limbs:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(overlay, pt1, pt2, (241, 102, 99), 2, cv2.LINE_AA)

        key_anchors = [
            (pose.neck, (56, 189, 248)),        # Cyan
            (pose.chest, (168, 85, 247)),      # Purple
            (pose.waist, (99, 102, 241)),      # Indigo
            (pose.left_shoulder, (236, 72, 153)), # Pink
            (pose.right_shoulder, (236, 72, 153)),
            (pose.left_wrist, (34, 197, 94)),   # Emerald
            (pose.right_wrist, (34, 197, 94)),
        ]

        for pt, color in key_anchors:
            center = (int(pt[0]), int(pt[1]))
            cv2.circle(overlay, center, 5, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, center, 8, color, 1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
