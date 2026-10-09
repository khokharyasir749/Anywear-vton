"""
Anywear VTO - Garment Processor Module
Automates garment isolation from e-commerce images (background removal, alpha masking, auto-cropping),
classifies garment categories (TOP vs BOTTOM), and extracts geometric anchor control points.
"""

import hashlib
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger("anywear-vton.garment")


@dataclass
class GarmentData:
    """Holds processed 4-channel garment data, classification, and geometric anchors."""
    url: str
    category: str  # "TOP" or "BOTTOM"
    image_bgra: np.ndarray  # (H, W, 4) with alpha channel
    mask: np.ndarray        # (H, W) uint8 binary mask
    width: int
    height: int
    anchors: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    aspect_ratio: float = 1.0


class GarmentProcessor:
    """
    Downloads, segments, and extracts anchor points from e-commerce product images.
    Caches processed garments for low-latency re-use.
    """

    def __init__(self, cache_dir: str = "cache_garments"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, GarmentData] = {}

    def process_url(
        self,
        url: str,
        category_override: Optional[str] = None
    ) -> Optional[GarmentData]:
        """
        Fetches an image from URL, strips background, extracts anchors, and caches result.
        """
        if not url:
            return None

        cache_key = f"{url}_{category_override or 'auto'}"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        try:
            # 1. Download image
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch garment image: HTTP {resp.status_code}")
                return None

            image_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error("Could not decode downloaded image bytes.")
                return None

            # 2. Isolate garment and create 4-channel BGRA with alpha mask
            bgra, mask = self._isolate_garment(img)

            # 3. Crop tightly to garment bounding box
            cropped_bgra, cropped_mask = self._auto_crop_garment(bgra, mask)

            gh, gw = cropped_bgra.shape[:2]
            aspect_ratio = gh / float(max(1, gw))

            # 4. Classify category (TOP vs BOTTOM)
            category = category_override.upper() if category_override else self._classify_category(aspect_ratio)

            # 5. Extract geometric control anchors
            anchors = self._extract_anchors(cropped_mask, gw, gh, category)

            garment_data = GarmentData(
                url=url,
                category=category,
                image_bgra=cropped_bgra,
                mask=cropped_mask,
                width=gw,
                height=gh,
                anchors=anchors,
                aspect_ratio=aspect_ratio
            )

            # Store in cache
            self.memory_cache[cache_key] = garment_data
            logger.info(
                f"Garment processed successfully: {category} ({gw}x{gh}), anchors={list(anchors.keys())}"
            )
            return garment_data

        except Exception as e:
            logger.error(f"Error processing garment URL {url}: {e}", exc_info=True)
            return None

    def _isolate_garment(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes solid white/gray studio background and returns BGRA with alpha mask.
        """
        # Case A: Image already contains an alpha channel
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            # If alpha is meaningful (not all 255), use it directly
            if np.any(alpha < 240):
                mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)[1]
                return img, mask
            else:
                bgr = img[:, :, :3]
        elif len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img

        h, w = bgr.shape[:2]

        # Case B: E-Commerce white/studio background removal
        # Sample the 4 corners to detect the ambient background color
        corners = [
            bgr[0:15, 0:15],
            bgr[0:15, w-15:w],
            bgr[h-15:h, 0:15],
            bgr[h-15:h, w-15:w]
        ]
        corner_means = [np.mean(c, axis=(0, 1)) for c in corners]
        bg_color = np.median(corner_means, axis=0) # [B, G, R]

        # Calculate Euclidean color distance from background color
        diff = np.linalg.norm(bgr.astype(np.float32) - bg_color.astype(np.float32), axis=2)

        # Standard threshold: pixels close to background color become transparent
        thresh_val = 26.0 if np.mean(bg_color) > 200 else 35.0
        mask = (diff > thresh_val).astype(np.uint8) * 255

        # Morphological filtering to clean noise and fill small internal holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Smooth edges with slight feathering
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Construct BGRA output
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask

        return bgra, mask

    def _auto_crop_garment(self, bgra: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crops image tightly around the largest garment contour with a small safety margin."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bgra, mask

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add 6px padding
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(bgra.shape[1], x + w + pad)
        y2 = min(bgra.shape[0], y + h + pad)

        return bgra[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _classify_category(self, aspect_ratio: float) -> str:
        """
        Classifies garment based on aspect ratio (H / W):
        Tall & narrow (> 1.35) -> BOTTOM (pants, jeans, trousers)
        Wider / square (<= 1.35) -> TOP (shirts, t-shirts, jackets)
        """
        return "BOTTOM" if aspect_ratio > 1.35 else "TOP"

    def _extract_anchors(
        self, mask: np.ndarray, gw: int, gh: int, category: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Extracts strategic geometric anchors on the 2D garment for warping.
        Provides 16+ key anatomical anchor zones for Delaunay piecewise affine warping.
        """
        anchors = {}

        if category == "TOP":
            # 1. Neckline & Collar Anchors
            anchors["collar_center"] = (float(gw * 0.50), float(gh * 0.07))
            anchors["collar_left"] = (float(gw * 0.38), float(gh * 0.05))
            anchors["collar_right"] = (float(gw * 0.62), float(gh * 0.05))

            # 2. Shoulder Seams
            anchors["left_shoulder"] = (float(gw * 0.20), float(gh * 0.13))
            anchors["right_shoulder"] = (float(gw * 0.80), float(gh * 0.13))

            # 3. Outer Sleeves & Underarms
            anchors["left_sleeve"] = (float(gw * 0.04), float(gh * 0.30))
            anchors["right_sleeve"] = (float(gw * 0.96), float(gh * 0.30))
            anchors["left_armpit"] = (float(gw * 0.23), float(gh * 0.38))
            anchors["right_armpit"] = (float(gw * 0.77), float(gh * 0.38))

            # 4. Chest & Ribs
            anchors["chest_center"] = (float(gw * 0.50), float(gh * 0.36))
            anchors["left_rib"] = (float(gw * 0.24), float(gh * 0.62))
            anchors["right_rib"] = (float(gw * 0.76), float(gh * 0.62))
            anchors["left_mid"] = anchors["left_rib"]
            anchors["right_mid"] = anchors["right_rib"]

            # 5. Waist & Bottom Hem
            anchors["waist_center"] = (float(gw * 0.50), float(gh * 0.72))
            anchors["left_waist"] = (float(gw * 0.23), float(gh * 0.86))
            anchors["right_waist"] = (float(gw * 0.77), float(gh * 0.86))
            anchors["left_hem"] = (float(gw * 0.22), float(gh * 0.96))
            anchors["right_hem"] = (float(gw * 0.78), float(gh * 0.96))
            anchors["hem_center"] = (float(gw * 0.50), float(gh * 0.97))

        else: # BOTTOM
            # 1. Waistband
            anchors["waist_center"] = (float(gw * 0.50), float(gh * 0.06))
            anchors["left_waist"] = (float(gw * 0.18), float(gh * 0.06))
            anchors["right_waist"] = (float(gw * 0.82), float(gh * 0.06))

            # 2. Pelvis & Crotch
            anchors["crotch_center"] = (float(gw * 0.50), float(gh * 0.38))
            anchors["left_thigh_outer"] = (float(gw * 0.14), float(gh * 0.38))
            anchors["right_thigh_outer"] = (float(gw * 0.86), float(gh * 0.38))

            # 3. Knees
            anchors["left_knee"] = (float(gw * 0.30), float(gh * 0.68))
            anchors["right_knee"] = (float(gw * 0.70), float(gh * 0.68))
            anchors["left_knee_outer"] = (float(gw * 0.16), float(gh * 0.68))
            anchors["right_knee_outer"] = (float(gw * 0.84), float(gh * 0.68))

            # 4. Ankles & Hem
            anchors["left_ankle"] = (float(gw * 0.28), float(gh * 0.96))
            anchors["right_ankle"] = (float(gw * 0.72), float(gh * 0.96))
            anchors["left_ankle_outer"] = (float(gw * 0.18), float(gh * 0.96))
            anchors["right_ankle_outer"] = (float(gw * 0.82), float(gh * 0.96))

        return anchors
