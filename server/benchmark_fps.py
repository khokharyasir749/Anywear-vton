"""
Benchmark: Real-Time FPS and Per-Frame Latency Benchmark
Tests live pipeline throughput with active garment rendering and pose estimation.
"""

import time
import cv2
import numpy as np

from pose_detector import PoseDetector
from garment_processor import GarmentProcessor, GarmentData
from vton_engine import VTONEngine
from body_parser import BodyParser
from depth_shading import DepthShadingEngine

def benchmark_live_pipeline():
    print("=== LIVE PIPELINE 35+ FPS & LATENCY BENCHMARK ===")
    
    # 1. Initialize components
    pose_detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
    vton_engine = VTONEngine()
    
    # 2. Synthetic realistic human test frame (480x360 standard streaming canvas)
    h, w = 360, 480
    frame = np.full((h, w, 3), (180, 190, 200), dtype=np.uint8)
    # Draw simple torso/head in frame to create realistic texture
    cv2.circle(frame, (240, 70), 30, (140, 170, 210), -1) # head
    cv2.rectangle(frame, (180, 110), (300, 280), (100, 110, 140), -1) # torso
    
    # 3. Create mock top garment
    gh, gw = 280, 260
    garment_bgra = np.zeros((gh, gw, 4), dtype=np.uint8)
    cv2.rectangle(garment_bgra, (20, 20), (gw - 20, gh - 20), (220, 80, 50, 255), -1)
    
    # Extract anchors
    processor = GarmentProcessor()
    mask = (garment_bgra[:, :, 3] > 20).astype(np.uint8) * 255
    anchors = processor._extract_anchors(mask, gw, gh, "TOP")
    top_garment = GarmentData(
        url="test://mock_shirt",
        category="TOP",
        image_bgra=garment_bgra,
        mask=mask,
        width=gw,
        height=gh,
        anchors=anchors,
        aspect_ratio=gh / gw
    )
    
    # Warm-up run
    pose = pose_detector.detect(frame)
    _ = vton_engine.render(frame=frame, pose=pose, top_garment=top_garment, lighting_intensity=0.85)
    
    # Benchmark 50 consecutive frames
    num_frames = 50
    latencies = []
    
    t_start = time.perf_counter()
    for i in range(num_frames):
        t0 = time.perf_counter()
        
        # 1. Pose estimation (with alternate frame tracking & 480p internal downscaling)
        pose = pose_detector.detect(frame)
        
        # 2. VTON Engine render with 7-11px soft Gaussian feathering and collarbone Bezier neckline
        rendered = vton_engine.render(
            frame=frame,
            pose=pose,
            top_garment=top_garment,
            lighting_intensity=0.85,
            enable_physics=True
        )
        
        # 3. Fast JPEG encoding
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 65, int(cv2.IMWRITE_JPEG_OPTIMIZE), 0]
        _, _ = cv2.imencode(".jpg", rendered, encode_params)
        
        frame_time = (time.perf_counter() - t0) * 1000.0
        latencies.append(frame_time)
        
    total_time = time.perf_counter() - t_start
    avg_latency = sum(latencies) / len(latencies)
    effective_fps = num_frames / total_time
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    print(f"Total Frames Processed : {num_frames}")
    print(f"Average Frame Latency   : {avg_latency:.2f} ms")
    print(f"95th %ile Latency      : {p95_latency:.2f} ms")
    print(f"Effective Throughput   : {effective_fps:.1f} FPS")
    print(f"Target Performance     : {'PASSED [35+ FPS RESTORED]' if effective_fps >= 35.0 or avg_latency < 28.0 else 'FAILED'}")
    print("=" * 50)

if __name__ == "__main__":
    benchmark_live_pipeline()
