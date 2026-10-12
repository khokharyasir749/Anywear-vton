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
        Creates preservation masks for Face, Neck, Wrists, Hands, and Forearms.
        Forearms are modeled as capsules between elbows and wrists to ensure sleeves
        are tucked cleanly behind arms resting across the body.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Face & Head region (Nose + Chin coverage)
        nose = pose.landmarks.get("nose", (w * 0.5, h * 0.2))
        nose_x, nose_y = int(nose[0]), int(nose[1])
        head_r = int(max(25, pose.shoulder_width * 0.35))
        cv2.circle(mask, (nose_x, nose_y), head_r, 255, -1)

        # Chin / Jawline protection (lower half of face)
        chin_y = int(nose_y + head_r * 0.65)
        cv2.ellipse(
            mask,
            (nose_x, chin_y),
            (int(head_r * 0.70), int(head_r * 0.40)),
            0, 0, 360, 255, -1
        )

        # 2. Neck skin preservation contour (strictly above the collar neckline, between chin and collarbone)
        neck_x = int(pose.neck[0])
        neck_center_y = int(pose.neck[1] - pose.torso_height * 0.09)
        cv2.ellipse(
            mask,
            (neck_x, neck_center_y),
            (int(pose.shoulder_width * 0.15), int(pose.torso_height * 0.08)),
            0, 0, 360, 255, -1
        )

        # 3. Forearm Capsules (Elbow -> Wrist)
        # Prevents virtual shirts from drawing over arms folded or resting on chest/waist
        arm_thickness = int(max(16, pose.shoulder_width * 0.14))
        for elbow, wrist in [(pose.left_elbow, pose.left_wrist), (pose.right_elbow, pose.right_wrist)]:
            ex, ey = int(elbow[0]), int(elbow[1])
            wx, wy = int(wrist[0]), int(wrist[1])
            if ex > 0 and ey > 0 and wx > 0 and wy > 0:
                cv2.line(mask, (ex, ey), (wx, wy), 255, thickness=arm_thickness)

        # 4. Hands & Wrists
        hand_r = int(max(18, pose.shoulder_width * 0.16))
        for joint in [pose.left_wrist, pose.right_wrist]:
            jx, jy = int(joint[0]), int(joint[1])
            if 0 <= jx < w and 0 <= jy < h:
                cv2.circle(mask, (jx, jy), hand_r, 255, -1)

        return cv2.GaussianBlur(mask, (5, 5), 0)

    def _generate_upper_torso_mask(
        self, w: int, h: int, pose: PoseData, skin_preserve: np.ndarray
    ) -> np.ndarray:
        """
        Constructs the Upper Torso replacement mask (clothing area).
        Fills the core torso region solidly (255) to completely replace original clothing
        rather than overlaying transparently.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        l_sh = pose.left_shoulder
        r_sh = pose.right_shoulder
        neck = pose.neck
        waist = pose.waist

        sh_w = pose.shoulder_width
        torso_h = pose.torso_height

        # Direction vector along shoulders
        dx = (r_sh[0] - l_sh[0]) / max(1.0, sh_w)
        dy = (r_sh[1] - l_sh[1]) / max(1.0, sh_w)
        down_x = -dy
        down_y = dx
        if down_y < 0:
            down_x, down_y = -down_x, -down_y

        sleeve_pad = sh_w * 0.28

        # Collarbone / Clavicle anatomical landmarks
        collar_left_base = (
            l_sh[0] + dx * (sh_w * 0.22) - down_x * (torso_h * 0.02),
            l_sh[1] + dy * (sh_w * 0.22) - down_y * (torso_h * 0.02)
        )
        collar_right_base = (
            r_sh[0] - dx * (sh_w * 0.22) - down_x * (torso_h * 0.02),
            r_sh[1] - dy * (sh_w * 0.22) - down_y * (torso_h * 0.02)
        )
        collar_center_notch = (
            neck[0] + down_x * (torso_h * 0.03),
            neck[1] + down_y * (torso_h * 0.03)
        )

        # Smooth Quadratic Bezier curve across collarbones (P0: left base -> P1: center dip -> P2: right base)
        # Eliminates rigid straight-line polygon cuts and molds naturally to the human neckline
        neck_curve_pts = []
        for t in np.linspace(0.0, 1.0, 15):
            bx = ((1.0 - t) ** 2) * collar_left_base[0] + 2.0 * (1.0 - t) * t * collar_center_notch[0] + (t ** 2) * collar_right_base[0]
            by = ((1.0 - t) ** 2) * collar_left_base[1] + 2.0 * (1.0 - t) * t * collar_center_notch[1] + (t ** 2) * collar_right_base[1]
            neck_curve_pts.append([int(bx), int(by)])

        # Assemble full upper torso contour with smooth collarbone neckline
        poly_pts = [
            [int(l_sh[0] - dx * sleeve_pad), int(l_sh[1] - dy * sleeve_pad - torso_h * 0.06)] # Left outer shoulder
        ]
        poly_pts.extend(neck_curve_pts)
        poly_pts.extend([
            [int(r_sh[0] + dx * sleeve_pad), int(r_sh[1] + dy * sleeve_pad - torso_h * 0.06)], # Right outer shoulder
            [int(r_sh[0] + dx * (sleeve_pad * 1.35)), int(r_sh[1] + torso_h * 0.52)],         # Right sleeve
            [int(r_sh[0] + dx * (sh_w * 0.12)), int(waist[1] + torso_h * 0.15)],              # Right waist hem
            [int(waist[0] + down_x * (torso_h * 0.16)), int(waist[1] + down_y * (torso_h * 0.16))], # Center hem
            [int(l_sh[0] - dx * (sh_w * 0.12)), int(waist[1] + torso_h * 0.15)],              # Left waist hem
            [int(l_sh[0] - dx * (sleeve_pad * 1.35)), int(l_sh[1] + torso_h * 0.52)],         # Left sleeve
        ])

        cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 255)

        # Subtract sensitive skin zones (chin, neck, forearms, hands)
        mask = cv2.subtract(mask, skin_preserve)

        # Smooth edges with 9px Gaussian feathering so mask perimeter melts naturally
        return cv2.GaussianBlur(mask, (9, 9), 2.5)

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
