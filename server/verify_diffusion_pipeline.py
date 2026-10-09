"""
End-to-End Verification Test for Neural Diffusion Pipeline (CatVTON / IDM-VTON)
Validates:
1. Dynamic hardware probing (CUDA / GPU VRAM / CPU fallback).
2. Diffusion engine initialization & Gradio client connectivity.
3. High-resolution try-on execution (cloud CatVTON or high-res seamless neural inpainting fallback).
4. REST API endpoints (/api/hardware, /api/tryon/neural).
"""

import time
import numpy as np
import cv2

from diffusion_engine import DiffusionVTONEngine
from garment_processor import GarmentData, GarmentProcessor


def create_test_human_image(w=640, h=480) -> np.ndarray:
    """Generates synthetic human test image."""
    img = np.full((h, w, 3), (210, 220, 230), dtype=np.uint8) # Studio background

    # Person body (head, neck, torso)
    head_cx, head_cy = w // 2, int(h * 0.22)
    cv2.circle(img, (head_cx, head_cy), 45, (185, 205, 235), -1) # Face/head skin

    # Neck
    cv2.rectangle(img, (head_cx - 15, head_cy + 35), (head_cx + 15, head_cy + 75), (180, 200, 230), -1)

    # Torso (existing shirt)
    torso_top = head_cy + 65
    torso_pts = np.array([
        [head_cx - 90, torso_top + 20],
        [head_cx + 90, torso_top + 20],
        [head_cx + 75, h - 30],
        [head_cx - 75, h - 30]
    ], dtype=np.int32)
    cv2.fillPoly(img, [torso_pts], (70, 70, 90)) # Dark shirt

    # Add real-world fold luminance
    for y in range(torso_top + 40, h - 60, 25):
        cv2.line(img, (head_cx - 50, y), (head_cx + 50, y + 5), (45, 45, 60), 2)

    return img


def create_test_garment(gw=300, gh=340) -> np.ndarray:
    """Generates synthetic garment BGRA image."""
    bgra = np.zeros((gh, gw, 4), dtype=np.uint8)

    # Garment fabric color (emerald green)
    bgr_color = (80, 190, 50)
    cv2.rectangle(bgra, (int(gw * 0.2), int(gh * 0.1)), (int(gw * 0.8), int(gh * 0.95)), (*bgr_color, 255), -1)

    # Sleeves
    cv2.fillPoly(bgra, [np.array([[int(gw * 0.2), int(gh * 0.1)], [0, int(gh * 0.35)], [int(gw * 0.22), int(gh * 0.42)]])], (*bgr_color, 255))
    cv2.fillPoly(bgra, [np.array([[int(gw * 0.8), int(gh * 0.1)], [gw, int(gh * 0.35)], [int(gw * 0.78), int(gh * 0.42)]])], (*bgr_color, 255))

    return bgra


def test_hardware_probing():
    print("[1/4] Testing Dynamic Hardware Probing...")
    engine = DiffusionVTONEngine()
    status = engine.get_status()

    assert "hardware" in status, "Status must contain hardware info"
    hw = status["hardware"]
    print(f"  -> Detected Hardware: {hw.get('device_name')}")
    print(f"  -> CUDA Available: {hw.get('cuda_available')}")
    print(f"  -> VRAM (GB): {hw.get('vram_gb')}")
    print(f"  -> Active Diffusion Mode: {status.get('mode')}")
    print(f"  -> Target Cloud Space: {status.get('active_cloud_space')}")

    assert status.get("mode") in ("LOCAL_CUDA_DIFFUSERS", "CLOUD_GRADIO_CLIENT"), "Invalid mode selected"
    print("  -> PASSED: Hardware probing and mode selection verified.")


def test_base64_utilities():
    print("[2/4] Testing Base64 Encoding/Decoding Utilities...")
    engine = DiffusionVTONEngine()
    test_img = np.full((120, 160, 3), (120, 180, 240), dtype=np.uint8)

    data_url = engine.encode_bgr_to_base64_data_url(test_img)
    assert data_url.startswith("data:image/jpeg;base64,"), "Encoded string must be valid data URL"

    decoded_img = engine.decode_base64_to_bgr(data_url)
    assert decoded_img is not None, "Decoded image should not be None"
    assert decoded_img.shape == test_img.shape, f"Shape mismatch: {decoded_img.shape} vs {test_img.shape}"

    print("  -> PASSED: Base64 data URL encode and decode roundtrip verified.")


def test_diffusion_tryon_execution():
    print("[3/4] Testing Neural Diffusion Try-On Pipeline Execution...")
    engine = DiffusionVTONEngine()
    human = create_test_human_image()
    garment = create_test_garment()

    start_t = time.time()
    result_bgr, meta = engine.generate_photorealistic_tryon(human, garment, category="TOP")
    elapsed = time.time() - start_t

    assert result_bgr is not None, "Result image should not be None"
    assert result_bgr.shape == human.shape, f"Output shape {result_bgr.shape} must match input {human.shape}"
    assert "pipeline" in meta, "Metadata must describe active pipeline"
    assert "latency_sec" in meta, "Metadata must track latency"

    print(f"  -> Pipeline executed: {meta.get('pipeline')} in {meta.get('latency_sec')}s (Total: {elapsed:.2f}s)")
    print(f"  -> Output Resolution: {meta.get('resolution')}")

    # Verify that the garment color was synthesized into the chest/torso region
    chest_region = result_bgr[180:320, 260:380]
    # Check green channel dominance where green garment was worn
    mean_g = np.mean(chest_region[:, :, 1])
    mean_r = np.mean(chest_region[:, :, 2])
    assert mean_g > mean_r, "Garment color was successfully transferred to torso"

    print("  -> PASSED: Neural Virtual Try-On output synthesized cleanly.")


def test_fastapi_rest_endpoints():
    print("[4/4] Testing FastAPI REST Endpoints...")
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)

    # Test GET /api/hardware
    res_hw = client.get("/api/hardware")
    assert res_hw.status_code == 200, f"/api/hardware failed with {res_hw.status_code}"
    hw_json = res_hw.json()
    assert "mode" in hw_json, "Hardware response must contain mode"
    print(f"  -> /api/hardware response: mode={hw_json.get('mode')}")

    # Test POST /api/tryon/neural
    engine = DiffusionVTONEngine()
    human = create_test_human_image(320, 240)
    h_b64 = engine.encode_bgr_to_base64_data_url(human)

    # Create dummy garment URL that garment_processor will handle
    dummy_garment = create_test_garment(200, 220)
    g_b64 = engine.encode_bgr_to_base64_data_url(dummy_garment)

    # Save temp garment so garment_processor can load it
    import tempfile, os
    tmp_g_path = os.path.join(tempfile.gettempdir(), "test_g_neural.png")
    cv2.imwrite(tmp_g_path, dummy_garment)

    payload = {
        "human_image": h_b64,
        "garment_url": tmp_g_path,
        "slot": "TOP"
    }

    res_tryon = client.post("/api/tryon/neural", json=payload)
    assert res_tryon.status_code == 200, f"/api/tryon/neural failed with {res_tryon.status_code}"
    tryon_json = res_tryon.json()
    assert tryon_json.get("status") == "success", f"Neural tryon returned error: {tryon_json}"
    assert "result_image" in tryon_json, "Response must contain result_image"
    print(f"  -> /api/tryon/neural success: pipeline={tryon_json.get('pipeline')}, latency={tryon_json.get('latency_sec')}s")

    if os.path.exists(tmp_g_path):
        os.remove(tmp_g_path)

    print("  -> PASSED: REST Endpoints (/api/hardware, /api/tryon/neural) verified.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING NEURAL DIFFUSION (CatVTON / IDM-VTON) VERIFICATION SUITE")
    print("=" * 65)
    test_hardware_probing()
    test_base64_utilities()
    test_diffusion_tryon_execution()
    test_fastapi_rest_endpoints()
    print("=" * 65)
    print("ALL 4 NEURAL DIFFUSION PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
