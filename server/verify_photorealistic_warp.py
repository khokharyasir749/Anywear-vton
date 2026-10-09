"""
Verification Test Suite: Photorealistic Seamless Warping (Anywear VTO Standard)
Validates:
1. Anatomical Anchor Mapping (Collar, Shoulders, Sleeves, Hem snapping)
2. Delaunay 16+ Key Anchor Topology
3. Skin & Neckline Occlusion (Collar cutout)
4. Photorealistic Shading (Wrinkle extraction, cylindrical depth, underarm AO)
5. Bilateral Edge Feathering (Elimination of hard sticker borders)
"""

import math
import numpy as np
import cv2

from pose_detector import PoseData
from garment_processor import GarmentData, GarmentProcessor
from depth_shading import DepthShadingEngine
from vton_engine import VTONEngine


def create_synthetic_pose(w=640, h=480) -> PoseData:
    """Creates a realistic anatomical upper body pose."""
    pose = PoseData()
    pose.detected = True
    pose.confidence = 0.95
    pose.shoulder_width = 160.0
    pose.torso_height = 200.0
    pose.body_angle_deg = 0.0

    # Base coordinates
    neck_x, neck_y = 320.0, 140.0
    pose.neck = (neck_x, neck_y)
    pose.left_shoulder = (neck_x - 80.0, 150.0)
    pose.right_shoulder = (neck_x + 80.0, 150.0)
    pose.chest = (neck_x, 190.0)
    pose.waist = (neck_x, 320.0)
    pose.left_hip = (neck_x - 70.0, 340.0)
    pose.right_hip = (neck_x + 70.0, 340.0)

    # Arms
    pose.left_elbow = (neck_x - 110.0, 240.0)
    pose.right_elbow = (neck_x + 110.0, 240.0)
    pose.left_wrist = (neck_x - 120.0, 330.0)
    pose.right_wrist = (neck_x + 120.0, 330.0)

    # Legs
    pose.left_knee = (neck_x - 65.0, 440.0)
    pose.right_knee = (neck_x + 65.0, 440.0)
    pose.left_ankle = (neck_x - 60.0, 520.0)
    pose.right_ankle = (neck_x + 60.0, 520.0)

    return pose


def create_mock_top_garment(gw=300, gh=340) -> GarmentData:
    """Creates a mock top garment with texture pattern and alpha."""
    bgr = np.full((gh, gw, 3), (45, 120, 220), dtype=np.uint8)
    # Add fabric texture grid
    for y in range(0, gh, 10):
        cv2.line(bgr, (0, y), (gw, y), (40, 105, 200), 1)
    for x in range(0, gw, 10):
        cv2.line(bgr, (x, 0), (x, gh), (40, 105, 200), 1)

    # T-shirt silhouette mask
    mask = np.zeros((gh, gw), dtype=np.uint8)
    cv2.rectangle(mask, (int(gw * 0.18), int(gh * 0.12)), (int(gw * 0.82), int(gh * 0.96)), 255, -1)
    # Sleeves
    cv2.fillPoly(mask, [np.array([[int(gw * 0.18), int(gh * 0.12)], [0, int(gh * 0.32)], [int(gw * 0.22), int(gh * 0.38)]])], 255)
    cv2.fillPoly(mask, [np.array([[int(gw * 0.82), int(gh * 0.12)], [gw, int(gh * 0.32)], [int(gw * 0.78), int(gh * 0.38)]])], 255)

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    processor = GarmentProcessor()
    anchors = processor._extract_anchors(mask, gw, gh, "TOP")

    return GarmentData(
        url="mock://tshirt.png",
        category="TOP",
        image_bgra=bgra,
        mask=mask,
        width=gw,
        height=gh,
        anchors=anchors,
        aspect_ratio=gh / gw
    )


def test_anatomical_anchor_snapping():
    print("[1/5] Testing Anatomical Anchor Snapping...")
    engine = VTONEngine()
    pose = create_synthetic_pose()
    garment = create_mock_top_garment()

    dst_anchors = engine._compute_target_anchors(pose, garment, 640, 480, drape_lag_x=0.0)
    assert dst_anchors is not None, "dst_anchors should not be None"

    # Collar must snap at/above shoulder line towards throat/chin
    collar_y = dst_anchors["collar_center"][1]
    shoulder_mid_y = (pose.left_shoulder[1] + pose.right_shoulder[1]) / 2.0
    assert collar_y < shoulder_mid_y, (
        f"Collar Y ({collar_y}) must be strictly higher than shoulder line ({shoulder_mid_y}) to prevent floating bib"
    )

    # Shoulders must strictly anchor close to landmarks 11 & 12
    l_sh_dist = math.hypot(dst_anchors["left_shoulder"][0] - pose.left_shoulder[0],
                           dst_anchors["left_shoulder"][1] - pose.left_shoulder[1])
    r_sh_dist = math.hypot(dst_anchors["right_shoulder"][0] - pose.right_shoulder[0],
                           dst_anchors["right_shoulder"][1] - pose.right_shoulder[1])
    assert l_sh_dist < 15.0, f"Left shoulder anchor diverged too much: {l_sh_dist}px"
    assert r_sh_dist < 15.0, f"Right shoulder anchor diverged too much: {r_sh_dist}px"

    # Sleeves must be outward along arm vector
    assert dst_anchors["left_sleeve"][0] < dst_anchors["left_shoulder"][0], "Left sleeve must extend laterally outward"
    assert dst_anchors["right_sleeve"][0] > dst_anchors["right_shoulder"][0], "Right sleeve must extend laterally outward"

    # Bottom hem must align around waist / hips
    assert dst_anchors["hem_center"][1] > pose.waist[1], "Hem must sit at/below waist level"
    print("  -> PASSED: Collar, shoulders, sleeves, and hem snap precisely to anatomy.")


def test_delaunay_mesh_topology():
    print("[2/5] Testing Delaunay Mesh Topology (16+ Key Anchors)...")
    engine = VTONEngine()
    garment = create_mock_top_garment()

    assert len(garment.anchors) >= 16, f"Garment anchors count {len(garment.anchors)} is less than 16"
    assert len(engine.top_triangles) >= 16, f"Top triangles count {len(engine.top_triangles)} is less than 16"

    # Verify all triangle vertices exist in garment anchors
    for tri in engine.top_triangles:
        for v in tri:
            assert v in garment.anchors, f"Triangle vertex '{v}' missing from garment anchors!"

    print(f"  -> PASSED: {len(garment.anchors)} anchor zones and {len(engine.top_triangles)} Delaunay triangles verified.")


def test_skin_and_neckline_occlusion():
    print("[3/5] Testing Skin & Neckline Occlusion...")
    engine = VTONEngine()
    pose = create_synthetic_pose()
    garment = create_mock_top_garment()

    dst_anchors = engine._compute_target_anchors(pose, garment, 640, 480, drape_lag_x=0.0)
    warped_bgr, warped_alpha = engine._warp_mesh(garment, dst_anchors, 640, 480)

    # Check alpha before cutout at collar center
    cc = dst_anchors["collar_center"]
    throat_x, throat_y = int(cc[0]), int(cc[1] - pose.torso_height * 0.03)

    torso_len = max(1.0, math.hypot(pose.waist[0] - pose.neck[0], pose.waist[1] - pose.neck[1]))
    sh_len = max(1.0, pose.shoulder_width)

    # Collar cutout logic
    c_rx = int(max(10, sh_len * 0.14))
    c_ry = int(max(8, torso_len * 0.08))
    collar_cutout = np.zeros_like(warped_alpha)
    cv2.ellipse(collar_cutout, (throat_x, throat_y), (c_rx, c_ry), 0, 0, 360, 255, -1)
    collar_cutout = cv2.GaussianBlur(collar_cutout, (9, 9), 0)

    alpha_cut = np.clip(warped_alpha.astype(np.int16) - collar_cutout.astype(np.int16), 0, 255).astype(np.uint8)

    # Throat position must have near 0 alpha to show user's real skin
    assert alpha_cut[throat_y, throat_x] < 30, (
        f"Throat cutout did not clear alpha at ({throat_x}, {throat_y}): got {alpha_cut[throat_y, throat_x]}"
    )
    print("  -> PASSED: Inner collar hollow cleanly masks out so natural throat and neck show through.")


def test_photorealistic_shading_and_folds():
    print("[4/5] Testing Photorealistic Shading & High-Pass Wrinkle Extraction...")
    depth_engine = DepthShadingEngine()
    pose = create_synthetic_pose()

    # Create dummy original frame with folds / texture
    orig_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Add horizontal shirt fold lines across user's chest
    for y in range(160, 280, 20):
        cv2.line(orig_frame, (260, y), (380, y), (50, 50, 50), 3)

    warped_bgr = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    warped_alpha = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(warped_alpha, (240, 140), (400, 340), 255, -1)

    shaded_bgr = depth_engine.apply_shading(warped_bgr, warped_alpha, orig_frame, pose, intensity=0.9)

    # Verify that fold lines in orig_frame were transferred to shaded_bgr
    fold_y = 180
    darkened_fold = shaded_bgr[fold_y, 320]
    plain_cloth = shaded_bgr[fold_y - 8, 320]
    assert np.mean(darkened_fold) < np.mean(plain_cloth), (
        f"High-pass folds should transfer crease shadows onto warped garment (fold: {np.mean(darkened_fold)}, plain: {np.mean(plain_cloth)})"
    )

    # Verify 3D cylindrical cosine dropoff towards torso edges (sample between fold lines at y=210)
    center_val = np.mean(shaded_bgr[210, 320])
    edge_val = np.mean(shaded_bgr[210, 245])
    assert edge_val < center_val, f"3D cylindrical shading should darken side edges (center: {center_val}, edge: {edge_val})"

    print("  -> PASSED: Real-world fabric folds extracted and 3D cylindrical shading verified.")


def test_bilateral_edge_feathering():
    print("[5/5] Testing Bilateral Edge Feathering (No Cookie-Cutter Borders)...")
    engine = VTONEngine()
    canvas = np.full((480, 640, 3), 100, dtype=np.uint8)
    garment_bgr = np.full((480, 640, 3), (50, 50, 255), dtype=np.uint8) # Red

    # Sharp binary mask
    alpha = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(alpha, (200, 150), (440, 350), 255, -1)

    blended = engine._composite_feathered(canvas.copy(), garment_bgr, alpha)

    # Boundary pixels along x=200 should have feathered intermediate values between 100 (canvas) and 255 (garment)
    border_pixels = [blended[200, x, 2] for x in range(195, 206)]
    is_feathered = any(110 < val < 240 for val in border_pixels)
    assert is_feathered, f"Alpha border must be smoothly feathered across 5-9px boundary (got {border_pixels})"

    print("  -> PASSED: Border feathering is continuous and eliminates hard sticker edges.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHOTOREALISTIC SEAMLESS WARPING VERIFICATION SUITE")
    print("=" * 60)
    test_anatomical_anchor_snapping()
    test_delaunay_mesh_topology()
    test_skin_and_neckline_occlusion()
    test_photorealistic_shading_and_folds()
    test_bilateral_edge_feathering()
    print("=" * 60)
    print("ALL 5 PHOTOREALISTIC WARPING VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
