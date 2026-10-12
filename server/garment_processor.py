"""
Anywear VTO - Garment Processor Module
Automates garment isolation from e-commerce images (background removal, alpha masking, auto-cropping),
classifies garment categories (TOP vs BOTTOM), and extracts geometric anchor control points.
"""

import base64
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
            img = None
            raw_bytes = None

            # 1. Normalize protocol-relative URLs
            if url.startswith("//"):
                url = "https:" + url

            # 2. Case A: Base64 data URL
            if url.startswith("data:"):
                try:
                    base64_data = url.split(",", 1)[1] if "," in url else url
                    import urllib.parse
                    base64_data = urllib.parse.unquote(base64_data).strip()
                    # Fix padding if needed
                    padding = len(base64_data) % 4
                    if padding:
                        base64_data += "=" * (4 - padding)
                    raw_bytes = base64.b64decode(base64_data)
                    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                except Exception as b64_err:
                    logger.warning(f"Error parsing base64 data URL: {b64_err}")

            # 3. Case B: Local filesystem path (guard against Windows SMB network hangs on web URLs)
            elif not url.startswith(("http://", "https://", "ftp://")):
                if os.path.exists(url):
                    img = cv2.imread(url, cv2.IMREAD_UNCHANGED)

            # 4. Case C: Web HTTP/HTTPS URL
            else:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://www.daraz.pk/",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }

                # Fetch target URL
                resp = None
                try:
                    resp = requests.get(url, headers=headers, timeout=8)
                except Exception as req_err:
                    logger.warning(f"Initial fetch failed for {url[:80]}: {req_err}")

                # If failed or 404, check multiple CDN fallback variations
                if resp is None or resp.status_code != 200 or len(resp.content) < 500:
                    import re
                    candidates = []
                    # Fallback 1: Strip appended size & webp suffix to get base photo (e.g. .jpg_.webp -> .jpg)
                    base_stripped = re.sub(r'(\.(?:jpg|jpeg|png|webp))_.*$', r'\1', url, flags=re.IGNORECASE)
                    if base_stripped != url:
                        candidates.append(base_stripped)
                    # Fallback 2: Strip dimensions pattern
                    dim_stripped = re.sub(r'_\d+x\d+[^.]*(\.[a-zA-Z0-9]+)', r'\1', url)
                    dim_stripped = re.sub(r'_\.webp$', '', dim_stripped)
                    if dim_stripped not in candidates and dim_stripped != url:
                        candidates.append(dim_stripped)
                    # Fallback 3: Lazada/Daraz standard 720x720 or 400x400
                    if "lazcdn" in url or "daraz" in url or "alicdn" in url:
                        c_720 = re.sub(r'_\d+x\d+[^.]*', '_720x720q80', url)
                        if c_720 not in candidates and c_720 != url:
                            candidates.append(c_720)

                    for fb_url in candidates:
                        try:
                            logger.info(f"Retrying garment with fallback URL: {fb_url[:80]}")
                            fb_resp = requests.get(fb_url, headers=headers, timeout=8)
                            if fb_resp.status_code == 200 and len(fb_resp.content) > 500:
                                resp = fb_resp
                                break
                        except Exception:
                            pass

                if resp is not None and resp.status_code == 200 and len(resp.content) > 200:
                    raw_bytes = resp.content
                    image_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
                    img = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
                else:
                    status_info = f"HTTP {resp.status_code}" if resp else "Connection error"
                    logger.error(f"Failed to fetch garment image: {status_info} for {url[:90]}")

            # 5. Pillow decode fallback if OpenCV returns None (handles WebP, AVIF, TIFF, progressive JPEG, etc.)
            if img is None and raw_bytes is not None and len(raw_bytes) > 100:
                try:
                    from PIL import Image
                    pil_img = Image.open(io.BytesIO(raw_bytes))
                    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                        rgba = pil_img.convert("RGBA")
                        img = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
                    else:
                        rgb = pil_img.convert("RGB")
                        img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
                    logger.info(f"Decoded garment via Pillow fallback: {pil_img.format} {pil_img.size}")
                except Exception as pil_err:
                    logger.warning(f"Pillow decode fallback also failed: {pil_err}")

            if img is None:
                logger.error(f"Could not decode garment image from URL: {url[:100]}")
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
        Removes solid white/gray/beige studio background and returns BGRA with alpha mask.
        Supports multi-zone border sampling, perceptual CIELab distance, and morphological hole filling.
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

        # Case B: Multi-zone border background sampling (handles white, light gray, beige, cream)
        sw = max(6, int(w * 0.05))
        sh = max(6, int(h * 0.05))
        border_samples = [
            bgr[0:sh, 0:sw],
            bgr[0:sh, w - sw:w],
            bgr[h - sh:h, 0:sw],
            bgr[h - sh:h, w - sw:w],
            bgr[0:sh, w // 2 - sw // 2:w // 2 + sw // 2],
            bgr[h - sh:h, w // 2 - sw // 2:w // 2 + sw // 2],
            bgr[h // 2 - sh // 2:h // 2 + sh // 2, 0:sw],
            bgr[h // 2 - sh // 2:h // 2 + sh // 2, w - sw:w]
        ]
        sample_means = [np.mean(s, axis=(0, 1)) for s in border_samples if s.size > 0]
        bg_color = np.median(sample_means, axis=0) # [B, G, R]

        # Calculate Euclidean color distance in BGR
        diff_bgr = np.linalg.norm(bgr.astype(np.float32) - bg_color.astype(np.float32), axis=2)

        # Also calculate difference in CIELab for perceptual color separation (handles warm beige/cream studio bgs)
        lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2LAB)[0, 0]
        diff_lab = np.linalg.norm(lab_img.astype(np.float32) - bg_lab.astype(np.float32), axis=2)

        thresh_val = 24.0 if np.mean(bg_color) > 190 else 32.0
        mask = ((diff_bgr > thresh_val) | (diff_lab > 18.0)).astype(np.uint8) * 255

        # Morphological filtering to clean noise and fill small internal holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill all internal holes inside external contours so text/logos/prints on shirt are never hollowed out
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(mask, [c], -1, 255, -1)

        # Smooth edges with slight feathering
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)

        # Construct BGRA output
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask

        return bgra, mask

    def _auto_crop_garment(self, bgra: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops image tightly around the primary single garment.
        Disentangles multi-item catalog collages (e.g. Front + Back shirts displayed side-by-side)
        to isolate ONLY the prominent single front garment!
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bgra, mask

        h, w = mask.shape[:2]

        # Check for side-by-side dual-garment catalog collage (Front + Back)
        # In e-commerce photos (Daraz, Amazon, etc.), front & back are shown side by side.
        col_sum = np.sum(mask > 30, axis=0)
        mid_min = int(w * 0.32)
        mid_max = int(w * 0.68)

        peak_left = np.max(col_sum[:mid_min]) if mid_min > 0 else 0
        peak_right = np.max(col_sum[mid_max:]) if mid_max < w else 0

        # If there are two strong vertical mass peaks on left and right with a valley in between
        if peak_left > h * 0.25 and peak_right > h * 0.25:
            min_mid_idx = mid_min + int(np.argmin(col_sum[mid_min:mid_max]))
            valley_val = col_sum[min_mid_idx]
            max_peak = max(peak_left, peak_right)

            # Valley dip indicates two distinct garment objects (left = FRONT, right = BACK)
            if valley_val < 0.82 * max_peak:
                logger.info(
                    f"Detected side-by-side dual garment collage. Splitting at x={min_mid_idx} to isolate FRONT garment."
                )
                isolated_mask = mask.copy()
                isolated_mask[:, min_mid_idx:] = 0

                left_contours, _ = cv2.findContours(isolated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if left_contours:
                    chosen = max(left_contours, key=cv2.contourArea)
                    x, y, cw, ch = cv2.boundingRect(chosen)
                    pad = 6
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(min_mid_idx, x + cw + pad)
                    y2 = min(h, y + ch + pad)
                    return bgra[y1:y2, x1:x2], mask[y1:y2, x1:x2]

        # Standard single garment: take the largest primary contour
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)

        # Add 6px padding
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)

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
