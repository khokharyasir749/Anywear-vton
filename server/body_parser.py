"""
Anywear VTO - Body Parser & Garment Region Isolator
Performs selective segmentation to isolate Upper Torso (Tops) from Lower Torso / Legs (Bottoms),
preserving skin, neck, and hands so Tops and Bottoms can be swapped independently or simultaneously.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pose_detector import PoseData

logger = logging.getLogger("anywear-vton.parser")


class BodyParser:
    """
    Separates human body regions into Upper Torso (Top), Lower Torso (Bottom),
    and Critical Preservation Zones (Face, Neck, Collarbones, Hands).
    """

    def __init__(self):
        pass

    def get_body_masks(
        self,
        frame: np.ndarray,
        pose: PoseData
    ) -> Dict[str, np.ndarray]:
        """
        Generates individual binary/alpha masks for:
          - "upper_torso": Region for Shirts, Tees, Jackets
          - "lower_torso": Region for Pants, Jeans, Shorts
          - "skin_preserve": Sensitive regions that should never be overwritten (Face, Neck, Hands)
        """
        h, w = frame.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)

        if not pose.detected:
            return {
                "upper_torso": empty_mask.copy(),
                "lower_torso": empty_mask.copy(),
                "skin_preserve": empty_mask.copy()
            }

        # 1. Skin & Face Preservation Mask
        skin_preserve = self._generate_skin_preserve_mask(w, h, pose)

        # 2. Upper Torso Mask (Collar down to Beltline)
        upper_torso = self._generate_upper_torso_mask(w, h, pose, skin_preserve)

        # 3. Lower Torso & Legs Mask (Beltline down to Ankles)
        lower_torso = self._generate_lower_torso_mask(w, h, pose, skin_preserve)

        return {
            "upper_torso": upper_torso,
            "lower_torso": lower_torso,
            "skin_preserve": skin_preserve
        }

    def _generate_skin_preserve_mask(
        self, w: int, h: int, pose: PoseData
    ) -> np.ndarray:
        """
        Creates preservation masks for Face, Neck, Wrists, and Forearms.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        # Face & Head region
        nose_x, nose_y = int(pose.landmarks.get("nose", (w * 0.5, h * 0.2))[0]), \
                         int(pose.landmarks.get("nose", (w * 0.5, h * 0.2))[1])
        head_r = int(max(25, pose.shoulder_width * 0.35))
        cv2.circle(mask, (nose_x, nose_y), head_r, 255, -1)

        # Neck oval
        neck_x, neck_y = int(pose.neck[0]), int(pose.neck[1])
        cv2.ellipse(
            mask,
            (neck_x, neck_y),
            (int(pose.shoulder_width * 0.18), int(pose.torso_height * 0.14)),
            0, 0, 360, 255, -1
        )

        # Hands & Wrists
        hand_r = int(max(15, pose.shoulder_width * 0.14))
        for joint in [pose.left_wrist, pose.right_wrist]:
            jx, jy = int(joint[0]), int(joint[1])
            if 0 <= jx < w and 0 <= jy < h:
                cv2.circle(mask, (jx, jy), hand_r, 255, -1)

        return cv2.GaussianBlur(mask, (7, 7), 0)

    def _generate_upper_torso_mask(
        self, w: int, h: int, pose: PoseData, skin_preserve: np.ndarray
    ) -> np.ndarray:
        """
        Constructs the Upper Torso bounding polygon (Tops region).
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        l_sh = pose.left_shoulder
        r_sh = pose.right_shoulder
        neck = pose.neck
        waist = pose.waist

        sh_w = pose.shoulder_width
        torso_h = pose.torso_height

        # Overhang vector for sleeves
        dx = (r_sh[0] - l_sh[0]) / max(1.0, sh_w)
        dy = (r_sh[1] - l_sh[1]) / max(1.0, sh_w)
        sleeve_pad = sh_w * 0.18

        # Define polygon coordinates for upper torso
        poly_pts = np.array([
            [l_sh[0] - dx * sleeve_pad, l_sh[1] - dy * sleeve_pad - torso_h * 0.05], # Left shoulder top
            [neck[0], neck[1] - torso_h * 0.05],                                     # Collar notch
            [r_sh[0] + dx * sleeve_pad, r_sh[1] + dy * sleeve_pad - torso_h * 0.05], # Right shoulder top
            [r_sh[0] + dx * (sleeve_pad * 1.2), r_sh[1] + torso_h * 0.40],          # Right sleeve/armpit
            [r_sh[0] + dx * (sh_w * 0.08), waist[1] + torso_h * 0.08],               # Right waist hem
            [l_sh[0] - dx * (sh_w * 0.08), waist[1] + torso_h * 0.08],               # Left waist hem
            [l_sh[0] - dx * (sleeve_pad * 1.2), l_sh[1] + torso_h * 0.40],          # Left sleeve/armpit
        ], dtype=np.int32)

        cv2.fillPoly(mask, [poly_pts], 255)

        # Subtract sensitive skin zones (chin, hands crossing chest)
        mask = cv2.subtract(mask, skin_preserve)

        # Feather edges softly
        return cv2.GaussianBlur(mask, (7, 7), 0)

    def _generate_lower_torso_mask(
        self, w: int, h: int, pose: PoseData, skin_preserve: np.ndarray
    ) -> np.ndarray:
        """
        Constructs the Lower Torso and Legs bounding polygon (Bottoms region).
        Starts cleanly at the waistline / belt level to avoid overlapping the shirt.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        l_hip = pose.left_hip
        r_hip = pose.right_hip
        waist = pose.waist
        hip_w = pose.hip_width

        l_knee = pose.left_knee
        r_knee = pose.right_knee
        l_ank = pose.left_ankle
        r_ank = pose.right_ankle

        belt_y = waist[1] + pose.torso_height * 0.05
        leg_pad = max(18.0, hip_w * 0.28)

        # Left leg polygon
        pts_left_leg = np.array([
            [l_hip[0] - leg_pad, belt_y],
            [l_hip[0] + leg_pad * 0.5, belt_y],
            [l_knee[0] + leg_pad * 0.5, l_knee[1]],
            [l_ank[0] + leg_pad * 0.5, min(h - 1, l_ank[1] + 15)],
            [l_ank[0] - leg_pad * 0.5, min(h - 1, l_ank[1] + 15)],
            [l_knee[0] - leg_pad * 0.5, l_knee[1]],
        ], dtype=np.int32)

        # Right leg polygon
        pts_right_leg = np.array([
            [r_hip[0] - leg_pad * 0.5, belt_y],
            [r_hip[0] + leg_pad, belt_y],
            [r_knee[0] + leg_pad * 0.5, r_knee[1]],
            [r_ank[0] + leg_pad * 0.5, min(h - 1, r_ank[1] + 15)],
            [r_ank[0] - leg_pad * 0.5, min(h - 1, r_ank[1] + 15)],
            [r_knee[0] - leg_pad * 0.5, r_knee[1]],
        ], dtype=np.int32)

        # Waistband connecting block
        waist_pts = np.array([
            [l_hip[0] - leg_pad, belt_y],
            [r_hip[0] + leg_pad, belt_y],
            [r_hip[0] + leg_pad * 0.8, belt_y + pose.torso_height * 0.35],
            [l_hip[0] - leg_pad * 0.8, belt_y + pose.torso_height * 0.35],
        ], dtype=np.int32)

        cv2.fillPoly(mask, [waist_pts], 255)
        cv2.fillPoly(mask, [pts_left_leg], 255)
        cv2.fillPoly(mask, [pts_right_leg], 255)

        # Subtract skin
        mask = cv2.subtract(mask, skin_preserve)

        return cv2.GaussianBlur(mask, (7, 7), 0)
