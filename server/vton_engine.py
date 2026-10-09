"""
Anywear VTO - Advanced Virtual Try-On Engine (Step 3)
Features:
 - Multi-Garment Support: Independent Tops and Bottoms swapping
 - Realistic Depth Shading, Fabric Wrinkle Transfer, and Ambient Occlusion
 - Micro-Draping Physics (Inertia & Elastic Fabric Sway)
 - Body Parsing Isolation & Bilateral Edge Feathering
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from body_parser import BodyParser
from depth_shading import DepthShadingEngine
from garment_processor import GarmentData
from pose_detector import PoseData

logger = logging.getLogger("anywear-vton.engine")


class FabricPhysics:
    """Simulates 2D elastic spring-damper cloth drape inertia."""

    def __init__(self):
        self.hem_offset_x = 0.0
        self.hem_vel_x = 0.0
        self.prev_user_cx = None
        self.last_time = time.time()

    def update(self, user_cx: float) -> float:
        now = time.time()
        dt = np.clip(now - self.last_time, 0.016, 0.10)
        self.last_time = now

        if self.prev_user_cx is None:
            self.prev_user_cx = user_cx
            return 0.0

        # User lateral velocity (movement to left or right)
        user_vel_x = (user_cx - self.prev_user_cx) / dt
        self.prev_user_cx = user_cx

        # Inertia opposes user motion: target offset is -user_vel_x * inertia_factor
        target_lag = np.clip(-user_vel_x * 0.018, -25.0, 25.0)

        # Spring-damper physics
        spring_k = 18.0
        damping = 7.5
        acc = spring_k * (target_lag - self.hem_offset_x) - damping * self.hem_vel_x
        self.hem_vel_x += acc * dt
        self.hem_offset_x += self.hem_vel_x * dt

        return float(self.hem_offset_x)


class VTONEngine:
    """
    Advanced real-time Virtual Try-On Engine with depth-aware shading,
    fabric wrinkle synthesis, body segmentation isolation, and micro-draping physics.
    """

    def __init__(self):
        self.depth_shading = DepthShadingEngine()
        self.body_parser = BodyParser()
        self.physics = FabricPhysics()

        # Comprehensive Delaunay Triangular Mesh Topology for TOPS (18 Triangles)
        self.top_triangles = [
            # Collar & Neckline
            ("collar_center", "collar_left", "chest_center"),
            ("collar_center", "collar_right", "chest_center"),
            ("collar_left", "left_shoulder", "chest_center"),
            ("collar_right", "right_shoulder", "chest_center"),

            # Left Arm / Shoulder & Underarm
            ("left_shoulder", "left_sleeve", "left_armpit"),
            ("left_shoulder", "left_armpit", "chest_center"),

            # Right Arm / Shoulder & Underarm
            ("right_shoulder", "right_sleeve", "right_armpit"),
            ("right_shoulder", "right_armpit", "chest_center"),

            # Torso & Ribs
            ("left_armpit", "left_rib", "chest_center"),
            ("right_armpit", "right_rib", "chest_center"),
            ("left_rib", "waist_center", "chest_center"),
            ("right_rib", "waist_center", "chest_center"),

            # Beltline & Waist
            ("left_rib", "left_waist", "waist_center"),
            ("right_rib", "right_waist", "waist_center"),

            # Lower Hem
            ("left_waist", "left_hem", "hem_center"),
            ("left_waist", "hem_center", "waist_center"),
            ("right_waist", "right_hem", "hem_center"),
            ("right_waist", "waist_center", "hem_center"),
        ]

        # Comprehensive Delaunay Triangular Mesh Topology for BOTTOMS (12 Triangles)
        self.bottom_triangles = [
            # Waistband & Pelvis
            ("waist_center", "left_waist", "crotch_center"),
            ("waist_center", "right_waist", "crotch_center"),
            ("left_waist", "left_thigh_outer", "crotch_center"),
            ("right_waist", "right_thigh_outer", "crotch_center"),

            # Thigh to Knee
            ("left_thigh_outer", "left_knee_outer", "crotch_center"),
            ("crotch_center", "left_knee_outer", "left_knee"),
            ("right_thigh_outer", "right_knee_outer", "crotch_center"),
            ("crotch_center", "right_knee_outer", "right_knee"),

            # Knee to Ankle / Hem
            ("left_knee_outer", "left_ankle_outer", "left_knee"),
            ("left_knee", "left_ankle_outer", "left_ankle"),
            ("right_knee_outer", "right_ankle_outer", "right_knee"),
            ("right_knee", "right_ankle_outer", "right_ankle"),
        ]

    def render(
        self,
        frame: np.ndarray,
        pose: PoseData,
        top_garment: Optional[GarmentData] = None,
        bottom_garment: Optional[GarmentData] = None,
        garment: Optional[GarmentData] = None,
        lighting_intensity: float = 0.85,
        enable_physics: bool = True,
        lighting_adapt: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Renders active Top and/or active Bottom garments onto the user's frame
        with independent segmentation isolation and realistic depth-wrinkle synthesis.
        """
        if frame is None or not pose.detected:
            return frame

        # Backward compatibility support for legacy single-garment parameter
        if garment is not None:
            if garment.category == "BOTTOM" and bottom_garment is None:
                bottom_garment = garment
            elif top_garment is None:
                top_garment = garment

        if top_garment is None and bottom_garment is None:
            return frame

        h, w = frame.shape[:2]
        output = frame.copy()

        # Update micro-draping fabric inertia
        drape_lag_x = self.physics.update(pose.neck[0]) if enable_physics else 0.0

        # Extract body segment isolation masks
        body_masks = self.body_parser.get_body_masks(frame, pose)

        # 1. Render BOTTOM Garment (Pants / Jeans) if active
        if bottom_garment is not None:
            output = self._render_single_garment(
                canvas_frame=output,
                original_frame=frame,
                pose=pose,
                garment=bottom_garment,
                region_mask=body_masks["lower_torso"],
                lighting_intensity=lighting_intensity,
                drape_lag_x=drape_lag_x * 0.4
            )

        # 2. Render TOP Garment (Shirt / Jacket) if active
        if top_garment is not None:
            output = self._render_single_garment(
                canvas_frame=output,
                original_frame=frame,
                pose=pose,
                garment=top_garment,
                region_mask=body_masks["upper_torso"],
                lighting_intensity=lighting_intensity,
                drape_lag_x=drape_lag_x
            )

        return output

    def _render_single_garment(
        self,
        canvas_frame: np.ndarray,
        original_frame: np.ndarray,
        pose: PoseData,
        garment: GarmentData,
        region_mask: np.ndarray,
        lighting_intensity: float,
        drape_lag_x: float
    ) -> np.ndarray:
        """
        Executes mesh warping, depth shading, fold transfer, and feathered blending
        for a single garment against its isolated body region.
        """
        h, w = canvas_frame.shape[:2]

        # 1. Compute target destination anchors on user's body
        dst_anchors = self._compute_target_anchors(pose, garment, w, h, drape_lag_x)
        if not dst_anchors:
            return canvas_frame

        # 2. Piecewise Affine Mesh Warping across Delaunay topology
        warped_bgr, warped_alpha = self._warp_mesh(garment, dst_anchors, w, h)
        if np.max(warped_alpha) < 10:
            return canvas_frame

        torso_len = max(1.0, math.hypot(pose.waist[0] - pose.neck[0], pose.waist[1] - pose.neck[1]))
        sh_len = max(1.0, pose.shoulder_width)

        # 3. Inner Collar Hollow & Skin Neckline Occlusion (Snaps to neck contour)
        if garment.category == "TOP" and "collar_center" in dst_anchors:
            cc = dst_anchors["collar_center"]
            throat_x = int(cc[0])
            throat_y = int(cc[1] - torso_len * 0.02)
            c_rx = int(max(10, sh_len * 0.15))
            c_ry = int(max(8, torso_len * 0.09))

            collar_cutout = np.zeros_like(warped_alpha)
            cv2.ellipse(collar_cutout, (throat_x, throat_y), (c_rx, c_ry), 0, 0, 360, 255, -1)
            collar_cutout = cv2.GaussianBlur(collar_cutout, (7, 7), 0)

            # Punch out inner collar hole so user's natural throat and skin show through
            warped_alpha = np.clip(warped_alpha.astype(np.int16) - collar_cutout.astype(np.int16), 0, 255).astype(np.uint8)

        # 4. Construct Dynamic Torso Replacement Mask (Complete Replacement, Not Flat Overlay)
        # Binarize core garment region so virtual garment completely replaces user's clothing
        garment_core = (warped_alpha > 35).astype(np.uint8) * 255
        if region_mask is not None:
            # Region mask isolates clothing area (with forearms, hands, and neck subtracted)
            replacement_mask = cv2.bitwise_and(garment_core, region_mask)
        else:
            replacement_mask = garment_core

        if np.max(replacement_mask) < 10:
            return canvas_frame

        # Feather boundary slightly so Poisson / Multi-Band blending transitions seamlessly
        replacement_mask = cv2.GaussianBlur(replacement_mask, (5, 5), 0)

        # 5. Realistic Depth Shading, Live Fabric Wrinkles & Ambient Occlusion
        shaded_bgr = self.depth_shading.apply_shading(
            warped_bgr=warped_bgr,
            warped_alpha=replacement_mask,
            original_frame=original_frame,
            pose=pose,
            intensity=lighting_intensity
        )

        # 6. Foreground Chain / Necklace Detection (from original camera frame)
        acc_mask = None
        acc_roi = None
        if garment.category == "TOP":
            acc_mask, acc_roi = self.depth_shading.detect_foreground_accessories(
                original_frame=original_frame,
                pose=pose,
                upper_torso_mask=replacement_mask
            )

        # 7. Decart-Level Seamless Live Blending (Poisson Gradient & Multi-Band Pyramid)
        blended = self.depth_shading.seamless_poisson_blend(
            original_frame=canvas_frame,
            warped_garment_bgr=shaded_bgr,
            replacement_mask=replacement_mask,
            pose=pose,
            mode="auto"
        )

        # 8. Foreground Chain / Necklace Preservation Re-Compositing
        if acc_mask is not None and np.max(acc_mask) > 0:
            blended = self.depth_shading.recomposite_foreground_accessories(
                canvas=blended,
                original_frame=original_frame,
                accessory_mask=acc_mask,
                roi_box=acc_roi
            )

        return blended

    def _compute_target_anchors(
        self,
        pose: PoseData,
        garment: GarmentData,
        w: int,
        h: int,
        drape_lag_x: float
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Calculates user body anchor locations with micro-draping physics offsets.
        """
        dst = {}
        l_sh = pose.left_shoulder
        r_sh = pose.right_shoulder
        neck = pose.neck
        waist = pose.waist
        chest = pose.chest

        sh_vec_x = r_sh[0] - l_sh[0]
        sh_vec_y = r_sh[1] - l_sh[1]
        sh_len = max(1.0, math.hypot(sh_vec_x, sh_vec_y))
        u_sh_x = sh_vec_x / sh_len
        u_sh_y = sh_vec_y / sh_len

        down_x = -u_sh_y
        down_y = u_sh_x
        if down_y < 0:
            down_x, down_y = -down_x, -down_y

        torso_len = max(1.0, math.hypot(waist[0] - neck[0], waist[1] - neck[1]))

        if garment.category == "TOP":
            # 1. Throat / Suprasternal notch:
            # Collar center snaps directly to base of neck / throat (above shoulder line towards chin)
            throat_x = neck[0] - down_x * (torso_len * 0.08)
            throat_y = neck[1] - down_y * (torso_len * 0.08)
            dst["collar_center"] = (throat_x, throat_y)

            # Neckline scoop corners
            collar_notch_w = sh_len * 0.16
            dst["collar_left"] = (
                throat_x - u_sh_x * collar_notch_w - down_x * (torso_len * 0.02),
                throat_y - u_sh_y * collar_notch_w - down_y * (torso_len * 0.02)
            )
            dst["collar_right"] = (
                throat_x + u_sh_x * collar_notch_w - down_x * (torso_len * 0.02),
                throat_y + u_sh_y * collar_notch_w - down_y * (torso_len * 0.02)
            )

            # 2. Shoulder seams: strictly snapped to anatomical landmarks 11 & 12
            dst["left_shoulder"] = (
                l_sh[0] - u_sh_x * (sh_len * 0.04) - down_x * (torso_len * 0.02),
                l_sh[1] - u_sh_y * (sh_len * 0.04) - down_y * (torso_len * 0.02)
            )
            dst["right_shoulder"] = (
                r_sh[0] + u_sh_x * (sh_len * 0.04) - down_x * (torso_len * 0.02),
                r_sh[1] + u_sh_y * (sh_len * 0.04) - down_y * (torso_len * 0.02)
            )

            # 3. Sleeves & Armpits: follow user's upper arms scaled with arm thickness & angle
            l_elbow = pose.left_elbow if pose.left_elbow[0] > 0 else (l_sh[0] - sh_len * 0.35, l_sh[1] + torso_len * 0.5)
            r_elbow = pose.right_elbow if pose.right_elbow[0] > 0 else (r_sh[0] + sh_len * 0.35, r_sh[1] + torso_len * 0.5)

            # Left arm direction and thickness
            l_arm_dx = l_elbow[0] - l_sh[0]
            l_arm_dy = l_elbow[1] - l_sh[1]
            l_arm_len = max(1.0, math.hypot(l_arm_dx, l_arm_dy))
            l_arm_ux = l_arm_dx / l_arm_len
            l_arm_uy = l_arm_dy / l_arm_len

            # Right arm direction and thickness
            r_arm_dx = r_elbow[0] - r_sh[0]
            r_arm_dy = r_elbow[1] - r_sh[1]
            r_arm_len = max(1.0, math.hypot(r_arm_dx, r_arm_dy))
            r_arm_ux = r_arm_dx / r_arm_len
            r_arm_uy = r_arm_dy / r_arm_len

            arm_thick = max(16.0, sh_len * 0.16)

            # Sleeve tips along arm vector with lateral sleeve flare matching arm thickness
            dst["left_sleeve"] = (
                l_sh[0] + l_arm_ux * min(l_arm_len * 0.58, torso_len * 0.45) - l_arm_uy * (arm_thick * 0.40),
                l_sh[1] + l_arm_uy * min(l_arm_len * 0.58, torso_len * 0.45) + l_arm_ux * (arm_thick * 0.40)
            )
            dst["right_sleeve"] = (
                r_sh[0] + r_arm_ux * min(r_arm_len * 0.58, torso_len * 0.45) + r_arm_uy * (arm_thick * 0.40),
                r_sh[1] + r_arm_uy * min(r_arm_len * 0.58, torso_len * 0.45) - r_arm_ux * (arm_thick * 0.40)
            )

            # Underarm seams
            dst["left_armpit"] = (
                l_sh[0] + down_x * (torso_len * 0.28) - u_sh_x * (sh_len * 0.02),
                l_sh[1] + down_y * (torso_len * 0.28) - u_sh_y * (sh_len * 0.02)
            )
            dst["right_armpit"] = (
                r_sh[0] + down_x * (torso_len * 0.28) + u_sh_x * (sh_len * 0.02),
                r_sh[1] + down_y * (torso_len * 0.28) + u_sh_y * (sh_len * 0.02)
            )

            # 4. Chest & Ribs
            dst["chest_center"] = (chest[0], chest[1])
            rib_center = (
                neck[0] + down_x * (torso_len * 0.58),
                neck[1] + down_y * (torso_len * 0.58)
            )
            half_rib_w = sh_len * 0.48
            dst["left_rib"] = (
                rib_center[0] - u_sh_x * half_rib_w + drape_lag_x * 0.4,
                rib_center[1] - u_sh_y * half_rib_w
            )
            dst["right_rib"] = (
                rib_center[0] + u_sh_x * half_rib_w + drape_lag_x * 0.4,
                rib_center[1] + u_sh_y * half_rib_w
            )
            dst["left_mid"] = dst["left_rib"]
            dst["right_mid"] = dst["right_rib"]

            # 5. Waist & Lower Hem (anchored to landmarks 23/24)
            l_hip = pose.left_hip if pose.left_hip[0] > 0 else (waist[0] - sh_len * 0.40, waist[1])
            r_hip = pose.right_hip if pose.right_hip[0] > 0 else (waist[0] + sh_len * 0.40, waist[1])

            dst["waist_center"] = (waist[0] + drape_lag_x * 0.7, waist[1])
            dst["left_waist"] = (l_hip[0] - u_sh_x * (sh_len * 0.04) + drape_lag_x * 0.7, l_hip[1])
            dst["right_waist"] = (r_hip[0] + u_sh_x * (sh_len * 0.04) + drape_lag_x * 0.7, r_hip[1])

            hem_center = (
                waist[0] + down_x * (torso_len * 0.12) + drape_lag_x,
                waist[1] + down_y * (torso_len * 0.12)
            )
            dst["hem_center"] = hem_center
            dst["left_hem"] = (
                dst["left_waist"][0] + down_x * (torso_len * 0.12) + drape_lag_x,
                dst["left_waist"][1] + down_y * (torso_len * 0.12)
            )
            dst["right_hem"] = (
                dst["right_waist"][0] + down_x * (torso_len * 0.12) + drape_lag_x,
                dst["right_waist"][1] + down_y * (torso_len * 0.12)
            )

        else: # BOTTOM
            l_hip = pose.left_hip
            r_hip = pose.right_hip
            hip_len = max(1.0, math.hypot(r_hip[0] - l_hip[0], r_hip[1] - l_hip[1]))

            dst["waist_center"] = (waist[0], waist[1] + torso_len * 0.02)
            dst["left_waist"] = (l_hip[0] - u_sh_x * (hip_len * 0.12), l_hip[1] + torso_len * 0.02)
            dst["right_waist"] = (r_hip[0] + u_sh_x * (hip_len * 0.12), r_hip[1] + torso_len * 0.02)
            dst["crotch_center"] = (waist[0] + down_x * (hip_len * 0.48), waist[1] + down_y * (hip_len * 0.48))
            dst["left_thigh_outer"] = (l_hip[0] - u_sh_x * (hip_len * 0.22) + down_x * (hip_len * 0.40), l_hip[1] + down_y * (hip_len * 0.40))
            dst["right_thigh_outer"] = (r_hip[0] + u_sh_x * (hip_len * 0.22) + down_x * (hip_len * 0.40), r_hip[1] + down_y * (hip_len * 0.40))

            dst["left_knee"] = (pose.left_knee[0] + drape_lag_x * 0.5, pose.left_knee[1])
            dst["right_knee"] = (pose.right_knee[0] + drape_lag_x * 0.5, pose.right_knee[1])
            dst["left_knee_outer"] = (pose.left_knee[0] - u_sh_x * (hip_len * 0.18) + drape_lag_x * 0.5, pose.left_knee[1])
            dst["right_knee_outer"] = (pose.right_knee[0] + u_sh_x * (hip_len * 0.18) + drape_lag_x * 0.5, pose.right_knee[1])

            dst["left_ankle"] = (pose.left_ankle[0] + drape_lag_x, pose.left_ankle[1])
            dst["right_ankle"] = (pose.right_ankle[0] + drape_lag_x, pose.right_ankle[1])
            dst["left_ankle_outer"] = (pose.left_ankle[0] - u_sh_x * (hip_len * 0.15) + drape_lag_x, pose.left_ankle[1])
            dst["right_ankle_outer"] = (pose.right_ankle[0] + u_sh_x * (hip_len * 0.15) + drape_lag_x, pose.right_ankle[1])

        return dst

    def _warp_mesh(
        self,
        garment: GarmentData,
        dst_anchors: Dict[str, Tuple[float, float]],
        out_w: int,
        out_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs piecewise affine mesh warping."""
        warped_bgr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        warped_alpha = np.zeros((out_h, out_w), dtype=np.uint8)

        src_anchors = garment.anchors
        garment_bgr = garment.image_bgra[:, :, :3]
        garment_alpha = garment.image_bgra[:, :, 3]

        triangles = self.top_triangles if garment.category == "TOP" else self.bottom_triangles

        for p1, p2, p3 in triangles:
            if not (p1 in src_anchors and p2 in src_anchors and p3 in src_anchors):
                continue
            if not (p1 in dst_anchors and p2 in dst_anchors and p3 in dst_anchors):
                continue

            src_tri = np.array([src_anchors[p1], src_anchors[p2], src_anchors[p3]], dtype=np.float32)
            dst_tri = np.array([dst_anchors[p1], dst_anchors[p2], dst_anchors[p3]], dtype=np.float32)

            self._warp_triangle(garment_bgr, garment_alpha, warped_bgr, warped_alpha, src_tri, dst_tri)

        return warped_bgr, warped_alpha

    def _warp_triangle(
        self,
        src_bgr: np.ndarray,
        src_alpha: np.ndarray,
        dst_bgr: np.ndarray,
        dst_alpha: np.ndarray,
        src_tri: np.ndarray,
        dst_tri: np.ndarray
    ):
        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)

        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return
        if r2[0] >= dst_bgr.shape[1] or r2[1] >= dst_bgr.shape[0]:
            return

        src_tri_crop = [(src_tri[i][0] - r1[0], src_tri[i][1] - r1[1]) for i in range(3)]
        dst_tri_crop = [(dst_tri[i][0] - r2[0], dst_tri[i][1] - r2[1]) for i in range(3)]

        src_crop_bgr = src_bgr[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        src_crop_alpha = src_alpha[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        warp_mat = cv2.getAffineTransform(np.float32(src_tri_crop), np.float32(dst_tri_crop))

        warped_crop_bgr = cv2.warpAffine(
            src_crop_bgr, warp_mat, (r2[2], r2[3]),
            None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        warped_crop_alpha = cv2.warpAffine(
            src_crop_alpha, warp_mat, (r2[2], r2[3]),
            None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        tri_mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, np.int32(dst_tri_crop), 255, 16, 0)

        x1, y1 = max(0, r2[0]), max(0, r2[1])
        x2, y2 = min(dst_bgr.shape[1], r2[0] + r2[2]), min(dst_bgr.shape[0], r2[1] + r2[3])

        crop_x1 = x1 - r2[0]
        crop_y1 = y1 - r2[1]
        crop_x2 = crop_x1 + (x2 - x1)
        crop_y2 = crop_y1 + (y2 - y1)

        valid = (tri_mask[crop_y1:crop_y2, crop_x1:crop_x2] > 0) & \
                (warped_crop_alpha[crop_y1:crop_y2, crop_x1:crop_x2] > 0)

        dst_bgr[y1:y2, x1:x2][valid] = warped_crop_bgr[crop_y1:crop_y2, crop_x1:crop_x2][valid]
        dst_alpha[y1:y2, x1:x2][valid] = warped_crop_alpha[crop_y1:crop_y2, crop_x1:crop_x2][valid]

    def _composite_feathered(
        self,
        canvas: np.ndarray,
        garment_bgr: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Applies 5px to 9px Gaussian blur edge feathering along garment alpha borders
        to eradicate hard sticker/cookie-cutter edges and ensure seamless skin/canvas transition.
        """
        rx, ry, rw, rh = cv2.boundingRect(alpha)
        if rw <= 0 or rh <= 0:
            return canvas

        h, w = canvas.shape[:2]
        # Pad ROI by 8px so Gaussian feathering has space to fade out naturally
        pad = 8
        x1 = max(0, rx - pad)
        y1 = max(0, ry - pad)
        x2 = min(w, rx + rw + pad)
        y2 = min(h, ry + rh + pad)

        alpha_crop = alpha[y1:y2, x1:x2]
        
        # 9px Gaussian edge feathering along garment alpha borders
        feathered = cv2.GaussianBlur(alpha_crop, (9, 9), 2.5)
        norm_alpha = (feathered.astype(np.float32) / 255.0)[:, :, np.newaxis]

        garment_crop = garment_bgr[y1:y2, x1:x2].astype(np.float32)
        canvas_crop = canvas[y1:y2, x1:x2].astype(np.float32)

        blended = norm_alpha * garment_crop + (1.0 - norm_alpha) * canvas_crop
        canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return canvas
