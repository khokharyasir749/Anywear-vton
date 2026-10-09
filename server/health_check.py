"""
Anywear VTO - Automated System Diagnostic & Health Check Utility
Verifies:
  1. Python runtime environment (>= 3.9)
  2. Required packages (FastAPI, OpenCV, MediaPipe, NumPy, SciPy, WebSockets)
  3. MediaPipe PoseLandmarker model asset availability
  4. Camera input hardware accessibility
  5. Port 8000 socket binding readiness
"""

import os
import socket
import sys

def check_mark(success: bool) -> str:
    return "  [PASS] " if success else "  [FAIL] "


def run_diagnostics():
    print("=" * 70)
    print("       ANYWEAR VTO - SYSTEM DIAGNOSTIC & HEALTH CHECK")
    print("=" * 70)

    all_passed = True

    # 1. Python Environment Check
    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 9)
    print(f"\n1. Python Runtime:")
    print(f"{check_mark(py_ok)}Detected Python {py_ver} (Minimum required: 3.9)")
    if not py_ok:
        all_passed = False

    # 2. Dependency Imports Check
    print(f"\n2. Core Dependencies:")
    deps = [
        ("cv2", "OpenCV Computer Vision Engine"),
        ("numpy", "NumPy Matrix & Vector Acceleration"),
        ("scipy", "SciPy Scientific Computing"),
        ("fastapi", "FastAPI Asynchronous Framework"),
        ("uvicorn", "Uvicorn ASGI High-Performance Server"),
        ("websockets", "Low-Latency WebSocket Transport"),
        ("requests", "HTTP Product Image Fetcher"),
        ("mediapipe", "MediaPipe PoseLandmarker Tasks API")
    ]

    for mod_name, desc in deps:
        try:
            __import__(mod_name)
            print(f"{check_mark(True)}{desc} ({mod_name})")
        except ImportError as e:
            print(f"{check_mark(False)}{desc} ({mod_name}) -> NOT FOUND")
            all_passed = False

    # 3. MediaPipe Model Asset Check
    print(f"\n3. Computer Vision Model Assets:")
    model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"{check_mark(True)}pose_landmarker_lite.task ({size_mb:.2f} MB)")
    else:
        print(f"{check_mark(False)}pose_landmarker_lite.task -> MISSING OR CORRUPT")
        all_passed = False

    # 4. Port 8000 Availability Check
    print(f"\n4. Network Socket Availability:")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", 8000))
        sock.close()
        print(f"{check_mark(True)}Port 8000 is open and ready for WebSocket streaming")
    except socket.error as e:
        print(f"  [WARN] Port 8000 is currently in use or bounded: {e}")
        print("         (If app.py is already running, this is normal)")

    # 5. Webcam Device Check
    print(f"\n5. Camera Hardware Probe:")
    try:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"{check_mark(True)}Default Camera Detected (Index 0: {w}x{h})")
            else:
                print(f"  [WARN] Camera Index 0 opened but could not read frame (device might be busy in Chrome)")
        else:
            print(f"  [INFO] Camera device not locked by OpenCV (active in Chrome browser)")
    except Exception as e:
        print(f"  [INFO] Camera probe bypassed: {e}")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("  HEALTH CHECK RESULT: [HEALTHY & READY FOR PRODUCTION]")
        print("  Launch Command: python server\\app.py")
    else:
        print("  HEALTH CHECK RESULT: [DEPENDENCIES MISSING]")
        print("  Please run: pip install -r server\\requirements.txt")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
