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

        # 1. Extract Live Real-World Folds & Wrinkles within ROI (Bidirectional float high-pass)
        gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        low_pass = cv2.GaussianBlur(gray_roi, (17, 17), 0)
        high_pass = gray_roi.astype(np.float32) - low_pass.astype(np.float32)

        # Normalized fold luminance centered at 0.5 (creases < 0.5, highlights > 0.5)
        fold_norm = np.clip(0.5 + (high_pass / 128.0) * 0.85, 0.0, 1.0)
        fold_3ch = fold_norm[:, :, np.newaxis]

        # 2. Pegtop Soft-Light Blending + Multiply Shadow Transfer
        cloth_norm = cloth_roi.astype(np.float32) / 255.0
        cloth_sq = cloth_norm * cloth_norm
        soft_light_roi = (1.0 - 2.0 * fold_3ch) * cloth_sq + (2.0 * fold_3ch) * cloth_norm

        # Selective shadow multiply for deep creases
        multiply_shadow = cloth_norm * np.clip(fold_3ch * 1.15, 0.0, 1.0)
        deep_crease = (fold_3ch < 0.44).astype(np.float32)
        blended_folds = (1.0 - deep_crease * 0.45) * soft_light_roi + (deep_crease * 0.45) * multiply_shadow
        blended_folds_bgr = np.clip(blended_folds * 255.0, 0.0, 255.0)

        # 3. Torso 3D Cylindrical Curvature & Underarm Ambient Occlusion
        shading_roi = self._compute_roi_diffuse_and_ao(rx, ry, rw, rh, pose)
        shading_3ch = shading_roi[:, :, np.newaxis]

        # 4. Modulate Shading with user intensity
        mix_factor = float(np.clip(intensity * 0.72, 0.0, 1.0))
        shaded_base = (1.0 - mix_factor) * cloth_roi.astype(np.float32) + mix_factor * blended_folds_bgr
        ao_factor = (1.0 - intensity * 0.40) + (intensity * 0.40) * shading_3ch
        final_roi = np.clip(shaded_base * ao_factor, 0.0, 255.0).astype(np.uint8)

        # Write back shaded ROI to output buffer
        result_bgr = warped_bgr.copy()
        result_bgr[ry:ry + rh, rx:rx + rw] = final_roi
        return result_bgr

    def _compute_roi_diffuse_and_ao(
        self, rx: int, ry: int, rw: int, rh: int, pose: PoseData
    ) -> np.ndarray:
        """
        Torso 3D Cylindrical Gradient & Ambient Occlusion:
        Darkens side edges and underarm seams using a radial cosine shadow map
        to simulate realistic anatomical depth and curvature.
        """
        sh_cx = (pose.left_shoulder[0] + pose.right_shoulder[0]) * 0.5
        sh_half_w = max(20.0, pose.shoulder_width * 0.50)

        # Horizontal coordinates relative to torso center
        xs = np.linspace(rx - sh_cx, (rx + rw) - sh_cx, rw, dtype=np.float32)
        rel_x = np.clip(xs / sh_half_w, -1.0, 1.0)

        # 3D Cylindrical Cosine curvature: center is 1.0, sides curve away to 0.62
        thetas = rel_x * (math.pi * 0.44)
        diffuse_1d = 0.66 + 0.34 * np.cos(thetas)

        # Virtual key light orientation bias
        light_bias = 1.0 + 0.15 * rel_x * self.light_dir[0]
        diffuse_1d = np.clip(diffuse_1d * light_bias, 0.55, 1.15)

        # Tile vertically across ROI
        shading_roi = np.tile(diffuse_1d, (rh, 1))

        # Ambient Occlusion in underarms and side rib seams
        l_sh = pose.left_shoulder
        r_sh = pose.right_shoulder
        chest_y = pose.chest[1]
        r_ao = int(max(10, pose.shoulder_width * 0.18))

        l_armpit = (int(l_sh[0] + pose.shoulder_width * 0.05) - rx, int(chest_y) - ry)
        r_armpit = (int(r_sh[0] - pose.shoulder_width * 0.05) - rx, int(chest_y) - ry)

        ao_mask = np.ones((rh, rw), dtype=np.float32)
        if 0 <= l_armpit[0] < rw and 0 <= l_armpit[1] < rh:
            cv2.circle(ao_mask, l_armpit, r_ao, 0.70, -1)
        if 0 <= r_armpit[0] < rw and 0 <= r_armpit[1] < rh:
            cv2.circle(ao_mask, r_armpit, r_ao, 0.70, -1)

        ao_mask = cv2.GaussianBlur(ao_mask, (19, 19), 0)
        return shading_roi * ao_mask

    def detect_foreground_accessories(
        self,
        original_frame: np.ndarray,
        pose: PoseData,
        upper_torso_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detects high-contrast foreground accessories (necklaces, chains, pendants, ties,
        and lanyards) hanging down from the throat across the chest.
        Returns:
          - full_mask: Binary mask (0 or 255) of detected accessories on full canvas
          - (x1, y1, x2, y2): Bounding box of the accessory ROI
        """
        h, w = original_frame.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        if not pose.detected or pose.shoulder_width < 10:
            return empty_mask, (0, 0, 0, 0)

        neck = pose.neck
        chest = pose.chest
        sh_w = pose.shoulder_width
        torso_h = pose.torso_height

        # Define localized accessory search zone from base of neck down to mid-chest
        half_w = int(max(15, sh_w * 0.24))
        x1 = max(0, int(neck[0] - half_w))
        x2 = min(w, int(neck[0] + half_w))
        y1 = max(0, int(neck[1] - torso_h * 0.04))
        y2 = min(h, int(chest[1] + torso_h * 0.24))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return empty_mask, (0, 0, 0, 0)

        roi_bgr = original_frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Median background estimation (filters out thin chains & small pendants)
        # 31x31 median filter accurately reconstructs the underlying shirt fabric/skin
        ksize = 31 if min(roi_bgr.shape[:2]) >= 31 else 15
        bg_lum = cv2.medianBlur(gray, ksize)
        diff = cv2.absdiff(gray, bg_lum)

        # 2. High-contrast thresholding (metallic highlights or dark chain links)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # 3. Structural edge reinforcement via Canny
        edges = cv2.Canny(gray, 40, 130)
        combined = cv2.bitwise_or(thresh, edges)

        # 4. Morphological filtering: remove isolated noise, bridge chain links
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close)

        # 5. Connected components filter: retain structures hanging vertically from neck area
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered, connectivity=8)
        accessory_roi = np.zeros_like(filtered)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Accessories like chains & pendants are typically between 15 and 3500 pixels in this ROI
            if 12 <= area <= 4000:
                comp_y = stats[i, cv2.CC_STAT_TOP]
                comp_h = stats[i, cv2.CC_STAT_HEIGHT]
                # Keep components that extend down into the chest region
                if comp_y + comp_h >= 10:
                    accessory_roi[labels == i] = 255

        if np.max(accessory_roi) == 0:
            return empty_mask, (x1, y1, x2, y2)

        # Write into full canvas mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = accessory_roi

        # Constrain to torso region if provided
        if upper_torso_mask is not None:
            full_mask = cv2.bitwise_and(full_mask, upper_torso_mask)

        return full_mask, (x1, y1, x2, y2)

    def recomposite_foreground_accessories(
        self,
        canvas: np.ndarray,
        original_frame: np.ndarray,
        accessory_mask: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Re-composites detected high-contrast accessories (necklaces, chains, ties)
        from the user's camera feed directly on top of the virtual garment.
        """
        if accessory_mask is None or np.max(accessory_mask) == 0:
            return canvas

        if roi_box and roi_box[2] > roi_box[0] and roi_box[3] > roi_box[1]:
            x1, y1, x2, y2 = roi_box
            acc_crop = accessory_mask[y1:y2, x1:x2]
            if np.max(acc_crop) == 0:
                return canvas

            # Smooth alpha mask for anti-aliased chain edges
            alpha = cv2.GaussianBlur(acc_crop.astype(np.float32) / 255.0, (3, 3), 0.8)
            alpha_3ch = alpha[:, :, np.newaxis]

            orig_crop = original_frame[y1:y2, x1:x2].astype(np.float32)
            canv_crop = canvas[y1:y2, x1:x2].astype(np.float32)

            blended = (1.0 - alpha_3ch) * canv_crop + alpha_3ch * orig_crop
            canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            return canvas

        # Full frame compositing fallback
        alpha = cv2.GaussianBlur(accessory_mask.astype(np.float32) / 255.0, (3, 3), 0.8)[:, :, np.newaxis]
        orig_f = original_frame.astype(np.float32)
        canv_f = canvas.astype(np.float32)
        blended = (1.0 - alpha) * canv_f + alpha * orig_f
        return np.clip(blended, 0, 255).astype(np.uint8)

    def seamless_poisson_blend(
        self,
        original_frame: np.ndarray,
        warped_garment_bgr: np.ndarray,
        replacement_mask: np.ndarray,
        pose: PoseData,
        mode: str = "auto"
    ) -> np.ndarray:
        """
        Applies seamless blending between the virtual garment texture and the user's camera feed.
        Transfers ambient room lighting, shadows, and color tones onto the virtual garment
        so it looks physically present in the room without sticker artifacts.

        Modes:
          - "auto" / "laplacian": Multi-band Laplacian pyramid blending (vibrant, artifact-free, 12ms)
          - "poisson": cv2.seamlessClone Poisson gradient blending with automatic fallback
        """
        if replacement_mask is None or np.max(replacement_mask) < 10:
            return original_frame

        h, w = original_frame.shape[:2]
        rx, ry, rw, rh = cv2.boundingRect(replacement_mask)
        if rw <= 4 or rh <= 4:
            return original_frame

        # Pad bounding box to give the blending border room to dissolve naturally
        pad = 16
        x1 = max(0, rx - pad)
        y1 = max(0, ry - pad)
        x2 = min(w, rx + rw + pad)
        y2 = min(h, ry + rh + pad)

        src_roi = warped_garment_bgr[y1:y2, x1:x2]
        dst_roi = original_frame[y1:y2, x1:x2].copy()
        mask_roi = replacement_mask[y1:y2, x1:x2].copy()

        # Execute Poisson seamlessClone if explicitly requested
        if mode == "poisson":
            try:
                # Ensure mask has strictly zero borders to satisfy OpenCV Poisson preconditions
                m_clean = mask_roi.copy()
                m_clean[0, :] = 0
                m_clean[-1, :] = 0
                m_clean[:, 0] = 0
                m_clean[:, -1] = 0
                if np.max(m_clean) > 0:
                    center = (src_roi.shape[1] // 2, src_roi.shape[0] // 2)
                    cloned_roi = cv2.seamlessClone(src_roi, dst_roi, m_clean, center, cv2.NORMAL_CLONE)
                    result = original_frame.copy()
                    result[y1:y2, x1:x2] = cloned_roi
                    return result
            except Exception as e:
                logger.debug("cv2.seamlessClone fallback to Laplacian blending: %s", e)

        # Multi-band Laplacian pyramid blending (Default):
        # Solves boundary gradients across octave bands without DC color-drift
        blended_roi = self._laplacian_pyramid_blend(src_roi, dst_roi, mask_roi, levels=3)
        result = original_frame.copy()
        result[y1:y2, x1:x2] = blended_roi
        return result

    def _laplacian_pyramid_blend(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        mask: np.ndarray,
        levels: int = 3
    ) -> np.ndarray:
        """
        High-performance Multi-Band Laplacian Pyramid Blending.
        Decomposes garment and video into spatial frequency bands,
        blending boundaries seamlessly while preserving full garment color and fine fabric detail.
        """
        m = (mask.astype(np.float32) / 255.0)
        if len(m.shape) == 2:
            m = m[:, :, np.newaxis]

        # 1. Build Gaussian pyramids
        g_src = [src.astype(np.float32)]
        g_dst = [dst.astype(np.float32)]
        g_m = [m]

        for i in range(levels):
            g_src.append(cv2.pyrDown(g_src[-1]))
            g_dst.append(cv2.pyrDown(g_dst[-1]))
            g_m.append(cv2.pyrDown(g_m[-1]))

        # 2. Build Laplacian pyramids with explicit parent dimensions
        l_src = [g_src[-1]]
        l_dst = [g_dst[-1]]

        for i in range(levels, 0, -1):
            target_size = (g_src[i - 1].shape[1], g_src[i - 1].shape[0])
            up_s = cv2.pyrUp(g_src[i], dstsize=target_size)
            up_d = cv2.pyrUp(g_dst[i], dstsize=target_size)
            l_src.append(g_src[i - 1] - up_s)
            l_dst.append(g_dst[i - 1] - up_d)

        l_src.reverse()
        l_dst.reverse()

        # 3. Blend pyramids across frequency bands
        blended_pyr = []
        for i in range(levels + 1):
            cur_m = g_m[i]
            if len(cur_m.shape) == 2:
                cur_m = cur_m[:, :, np.newaxis]
            blended_band = cur_m * l_src[i] + (1.0 - cur_m) * l_dst[i]
            blended_pyr.append(blended_band)

        # 4. Collapse Laplacian pyramid back to full image
        curr = blended_pyr[levels]
        for i in range(levels - 1, -1, -1):
            target_size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
            curr = cv2.pyrUp(curr, dstsize=target_size) + blended_pyr[i]

        return np.clip(curr, 0, 255).astype(np.uint8)

