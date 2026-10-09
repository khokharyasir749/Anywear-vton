"""
Anywear VTO - Depth Shading & Fabric Wrinkle Engine (High Performance)
Generates pseudo-depth surface normals, transfers real-world fabric folds and wrinkles
from the user's live clothing onto the synthetic garment, and computes ambient occlusion.
Optimized with ROI bounding box processing for 35+ FPS performance.
"""

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from pose_detector import PoseData

logger = logging.getLogger("anywear-vton.shading")


class DepthShadingEngine:
    """
    Photorealistic lighting and fold synthesis engine.
    Optimized for real-time video streaming (>30 FPS) using ROI bounding-box acceleration.
    """

    def __init__(self):
        # Virtual key light vector: [Lx, Ly, Lz]
        light_vec = np.array([0.15, -0.45, 0.88], dtype=np.float32)
        self.light_dir = light_vec / np.linalg.norm(light_vec)

    def apply_shading(
        self,
        warped_bgr: np.ndarray,
        warped_alpha: np.ndarray,
        original_frame: np.ndarray,
        pose: PoseData,
        intensity: float = 0.85
    ) -> np.ndarray:
        """
        Applies pseudo-depth curvature, fabric wrinkle transfer,
        and ambient occlusion restricted to the garment's active ROI.
        """
        if not pose.detected or np.max(warped_alpha) < 10:
            return warped_bgr

        intensity = float(np.clip(intensity, 0.0, 1.0))
        if intensity < 0.02:
            return warped_bgr

        # Find active bounding box of the garment to avoid processing the whole canvas
        rx, ry, rw, rh = cv2.boundingRect(warped_alpha)
        if rw <= 0 or rh <= 0:
            return warped_bgr

        # Crop ROIs
        cloth_roi = warped_bgr[ry:ry + rh, rx:rx + rw]
        alpha_roi = warped_alpha[ry:ry + rh, rx:rx + rw]
        frame_roi = original_frame[ry:ry + rh, rx:rx + rw]

        # 1. Extract Live Real-World Folds within ROI
        gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        low_pass = cv2.GaussianBlur(gray_roi, (11, 11), 0)
        high_pass = cv2.subtract(gray_roi, low_pass)
        fold_roi = cv2.add(cv2.multiply(high_pass, 1.5), 128)
        fold_norm = fold_roi.astype(np.float32) / 255.0

        # 2. Fast Soft-Light Blending on ROI
        cloth_norm = cloth_roi.astype(np.float32) / 255.0
        fold_3ch = fold_norm[:, :, np.newaxis]

        # Pegtop Soft-Light: (1 - 2*b)*a*a + 2*b*a
        cloth_sq = cloth_norm * cloth_norm
        soft_light_roi = (1.0 - 2.0 * fold_3ch) * cloth_sq + (2.0 * fold_3ch) * cloth_norm
        soft_light_bgr = np.clip(soft_light_roi * 255.0, 0.0, 255.0)

        # 3. Torso Curvature & Ambient Occlusion within ROI
        shading_roi = self._compute_roi_diffuse_and_ao(rx, ry, rw, rh, pose)
        shading_3ch = shading_roi[:, :, np.newaxis]

        # 4. Modulate Shading with user intensity
        mix_factor = intensity * 0.60
        shaded_roi = (1.0 - mix_factor) * cloth_roi.astype(np.float32) + mix_factor * soft_light_bgr
        ao_factor = (1.0 - intensity * 0.35) + (intensity * 0.35) * shading_3ch
        final_roi = np.clip(shaded_roi * ao_factor, 0.0, 255.0).astype(np.uint8)

        # Write back shaded ROI to output buffer
        result_bgr = warped_bgr.copy()
        result_bgr[ry:ry + rh, rx:rx + rw] = final_roi
        return result_bgr

    def _compute_roi_diffuse_and_ao(
        self, rx: int, ry: int, rw: int, rh: int, pose: PoseData
    ) -> np.ndarray:
        """
        Fast 1D cylindrical shading computation mapped onto ROI.
        """
        sh_cx = (pose.left_shoulder[0] + pose.right_shoulder[0]) * 0.5
        sh_half_w = max(20.0, pose.shoulder_width * 0.52)

        # Horizontal coordinates relative to torso center
        xs = np.linspace(rx - sh_cx, (rx + rw) - sh_cx, rw, dtype=np.float32)
        rel_x = np.clip(xs / sh_half_w, -1.0, 1.0)
        thetas = rel_x * (math.pi / 3.0)

        nx = np.sin(thetas)
        nz = np.cos(thetas)
        diffuse_1d = nx * self.light_dir[0] + nz * self.light_dir[2]
        diffuse_1d = np.clip(diffuse_1d, 0.65, 1.10)

        # Tile vertically across ROI
        shading_roi = np.tile(diffuse_1d, (rh, 1))

        # Ambient Occlusion in armpit regions within ROI
        l_sh = pose.left_shoulder
        r_sh = pose.right_shoulder
        chest_y = pose.chest[1]
        r_ao = int(pose.shoulder_width * 0.16)

        l_armpit = (int(l_sh[0] + pose.shoulder_width * 0.08) - rx, int(chest_y) - ry)
        r_armpit = (int(r_sh[0] - pose.shoulder_width * 0.08) - rx, int(chest_y) - ry)

        if 0 <= l_armpit[0] < rw and 0 <= l_armpit[1] < rh:
            cv2.circle(shading_roi, l_armpit, r_ao, 0.78, -1)
        if 0 <= r_armpit[0] < rw and 0 <= r_armpit[1] < rh:
            cv2.circle(shading_roi, r_armpit, r_ao, 0.78, -1)

        return shading_roi
