"""
Anywear VTO - Production Real-Time Virtual Try-On Server (Step 3)
Asynchronous multi-threaded pipeline:
  - Dual Garment Slots (Simultaneous or Independent Tops & Bottoms)
  - Real-Time MediaPipe Pose Tracking with Adaptive EMA
  - Photorealistic Depth Shading & Live Fabric Wrinkle Transfer
  - Body Parsing Region Isolation & Micro-Draping Physics
Maintains 25+ FPS low-latency bi-directional WebSocket streaming.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import logging
import time
from typing import Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from garment_processor import GarmentData, GarmentProcessor
from pose_detector import PoseData, PoseDetector
from vton_engine import VTONEngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("anywear-vton.server")

app = FastAPI(
    title="Anywear VTO - Advanced Real-Time Virtual Try-On Engine",
    description="Step 3: Depth Shading, Wrinkle Synthesis, and Dual Garment Parsing",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vton_worker")
garment_processor = GarmentProcessor()


class ClientSession:
    """Per-connection session state supporting dual garment slots and lighting settings."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.pose_detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
            base_ema_alpha=0.55
        )
        self.vton_engine = VTONEngine()

        # Dual Garment Slots (Independent Top & Bottom Swapping)
        self.active_top: Optional[GarmentData] = None
        self.active_bottom: Optional[GarmentData] = None
        self.active_top_url: Optional[str] = None
        self.active_bottom_url: Optional[str] = None

        # Mode and Realism Settings
        self.category_mode: str = "AUTO"  # "AUTO", "TOP", "BOTTOM"
        self.view_mode: str = "ai"        # "ai", "skeleton", "camera"
        self.lighting_intensity: float = 0.85 # 0.0 to 1.0

        # Telemetry
        self.frames_received: int = 0
        self.frames_sent: int = 0
        self.fps: float = 0.0
        self.last_fps_time: float = time.time()
        self.fps_counter: int = 0
        self.last_pose_detected: bool = False

    def update_fps(self):
        self.frames_received += 1
        self.fps_counter += 1
        now = time.time()
        delta = now - self.last_fps_time
        if delta >= 1.0:
            self.fps = self.fps_counter / delta
            self.fps_counter = 0
            self.last_fps_time = now


def process_video_frame_sync(
    raw_bytes: bytes,
    session: ClientSession
) -> Optional[bytes]:
    """
    Synchronous CPU-bound pipeline running inside worker thread:
    1. Decode JPEG
    2. Pose estimation & EMA smoothing
    3. Dual Top/Bottom piecewise affine warping with depth shading & wrinkle transfer
    4. HUD telemetry overlay
    5. Fast JPEG encoding
    """
    try:
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        h, w = frame.shape[:2]
        session.update_fps()

        has_garments = (session.active_top is not None or session.active_bottom is not None)

        # 1. When NO garments are active and not in skeleton mode:
        # SKIP heavy depth shading, warping, and full image processing!
        # Returning None allows the client local webcam to play at fluid 30-60 FPS natively with 0ms latency.
        if not has_garments and session.view_mode != "skeleton":
            return None

        # 2. Pose Landmark Estimation
        pose = session.pose_detector.detect(frame)
        session.last_pose_detected = pose.detected

        # 3. Render Virtual Try-On Garments (Top and/or Bottom)
        if session.view_mode == "ai" and pose.detected and has_garments:
            frame = session.vton_engine.render(
                frame=frame,
                pose=pose,
                top_garment=session.active_top,
                bottom_garment=session.active_bottom,
                lighting_intensity=session.lighting_intensity,
                enable_physics=True
            )
        elif session.view_mode == "skeleton" and pose.detected:
            frame = session.pose_detector.draw_skeleton(frame, pose)

        # 4. Telemetry HUD Overlay
        frame = draw_hud(frame, session, pose)

        # 5. Ultra-fast JPEG Re-encoding (low compression overhead, skip 2-pass Huffman tree)
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), 65,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 0
        ]
        success, encoded_jpg = cv2.imencode(".jpg", frame, encode_params)
        if success:
            return encoded_jpg.tobytes()

    except Exception as e:
        logger.error(f"Error in sync frame processing: {e}", exc_info=True)

    return None


def draw_hud(frame: np.ndarray, session: ClientSession, pose: PoseData) -> np.ndarray:
    """Renders sleek top banner and dual garment slot thumbnails."""
    h, w = frame.shape[:2]

    # Banner Bar
    banner_h = 32
    cv2.rectangle(frame, (0, 0), (w, banner_h), (15, 17, 26), -1)
    cv2.line(frame, (0, banner_h), (w, banner_h), (50, 55, 75), 1)

    # Pose Status Dot
    dot_color = (80, 220, 80) if pose.detected else (50, 50, 220)
    cv2.circle(frame, (16, 16), 5, dot_color, -1, cv2.LINE_AA)

    # Status Label
    status_label = f"ANYWEAR VTO [STEP 3] | MODE: {session.category_mode}" if pose.detected else "ANYWEAR VTO [SEARCHING POSE]"
    cv2.putText(frame, status_label, (28, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS Counter
    fps_text = f"{session.fps:.1f} FPS"
    cv2.putText(frame, fps_text, (w - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 240, 160), 1, cv2.LINE_AA)

    # Timestamp
    now_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, now_str, (w - 75, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Dual Garment Slot Thumbnails (Upper-Right Corner)
    curr_y = banner_h + 10
    thumb_size = 58

    # Top Slot Thumbnail
    if session.active_top is not None:
        try:
            thumb = cv2.resize(session.active_top.image_bgra, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
            px = w - thumb_size - 10
            alpha = (thumb[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
            bgr = thumb[:, :, :3].astype(np.float32)
            bg = frame[curr_y:curr_y + thumb_size, px:px + thumb_size].astype(np.float32)
            frame[curr_y:curr_y + thumb_size, px:px + thumb_size] = (alpha * bgr + (1.0 - alpha) * bg).astype(np.uint8)
            cv2.rectangle(frame, (px - 1, curr_y - 1), (px + thumb_size + 1, curr_y + thumb_size + 1), (99, 102, 241), 1)
            cv2.putText(frame, "TOP", (px + 4, curr_y + thumb_size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 255, 100), 1, cv2.LINE_AA)
            curr_y += thumb_size + 8
        except Exception:
            pass

    # Bottom Slot Thumbnail
    if session.active_bottom is not None:
        try:
            thumb = cv2.resize(session.active_bottom.image_bgra, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
            px = w - thumb_size - 10
            alpha = (thumb[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
            bgr = thumb[:, :, :3].astype(np.float32)
            bg = frame[curr_y:curr_y + thumb_size, px:px + thumb_size].astype(np.float32)
            frame[curr_y:curr_y + thumb_size, px:px + thumb_size] = (alpha * bgr + (1.0 - alpha) * bg).astype(np.uint8)
            cv2.rectangle(frame, (px - 1, curr_y - 1), (px + thumb_size + 1, curr_y + thumb_size + 1), (236, 72, 153), 1)
            cv2.putText(frame, "BOTTOM", (px + 4, curr_y + thumb_size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 200, 100), 1, cv2.LINE_AA)
        except Exception:
            pass

    if not pose.detected:
        guide_msg = "Please stand in view of the camera"
        msg_size = cv2.getTextSize(guide_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        mx = (w - msg_size[0]) // 2
        my = int(h * 0.88)
        cv2.rectangle(frame, (mx - 10, my - 18), (mx + msg_size[0] + 10, my + 8), (15, 17, 26), -1)
        cv2.putText(frame, guide_msg, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 240), 1, cv2.LINE_AA)

    return frame


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Anywear VTO - Step 3 Production Engine",
        "features": [
            "Dual Independent Garment Slots (Tops & Bottoms)",
            "Dynamic Pseudo-Depth & Surface Normal Shading",
            "Real-World Live Fabric Wrinkle & Crease Transfer",
            "Body Segmentation Isolation & Micro-Draping Physics"
        ]
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    client_ip = websocket.client.host if websocket.client else "unknown"
    session = ClientSession(client_id=client_ip)
    logger.info(f"WebSocket client connected: {client_ip}")

    # Drop-oldest single-slot queue to eliminate buffer bloat & queue lag
    frame_queue = asyncio.Queue(maxsize=1)
    stop_event = asyncio.Event()

    async def worker_loop():
        """Processes only the freshest frame, discarding stale buffered frames."""
        while not stop_event.is_set():
            try:
                # Wait for next available frame
                raw_bytes = await frame_queue.get()
                frame_queue.task_done()

                # If a newer frame arrived while waiting, skip to the latest
                while not frame_queue.empty():
                    try:
                        raw_bytes = frame_queue.get_nowait()
                        frame_queue.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        break

                encoded_bytes = await loop.run_in_executor(
                    executor,
                    process_video_frame_sync,
                    raw_bytes,
                    session
                )
                if encoded_bytes:
                    await websocket.send_bytes(encoded_bytes)
                    session.frames_sent += 1
                else:
                    # Lightweight standby ACK so in-flight lock opens and telemetry updates
                    await websocket.send_text(json.dumps({
                        "type": "STANDBY_ACK",
                        "server_fps": round(session.fps, 1),
                        "pose_detected": session.last_pose_detected
                    }))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in video worker loop: {e}", exc_info=True)

    worker_task = asyncio.create_task(worker_loop())

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"]:
                raw_bytes = msg["bytes"]
                # Drop-oldest policy: discard any unconsumed older frame and keep newest
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        pass
                try:
                    frame_queue.put_nowait(raw_bytes)
                except asyncio.QueueFull:
                    pass

            elif "text" in msg and msg["text"]:
                raw_text = msg["text"]
                try:
                    payload = json.loads(raw_text)
                    event_type = payload.get("type")

                    if event_type == "SET_CLOTH":
                        cloth_url = payload.get("cloth_url")
                        requested_slot = payload.get("slot", session.category_mode).upper() # "AUTO", "TOP", "BOTTOM"

                        if cloth_url:
                            # Process and isolate garment in background
                            def _fetch():
                                cat = None if requested_slot == "AUTO" else requested_slot
                                return garment_processor.process_url(cloth_url, category_override=cat)

                            garment_data = await loop.run_in_executor(executor, _fetch)

                            if garment_data:
                                assigned_category = garment_data.category
                                if assigned_category == "TOP":
                                    session.active_top = garment_data
                                    session.active_top_url = cloth_url
                                else:
                                    session.active_bottom = garment_data
                                    session.active_bottom_url = cloth_url

                                await websocket.send_text(json.dumps({
                                    "type": "STATUS_ACK",
                                    "action": "SET_CLOTH",
                                    "status": "ready",
                                    "assigned_slot": assigned_category,
                                    "has_top": session.active_top is not None,
                                    "has_bottom": session.active_bottom is not None,
                                    "top_url": session.active_top_url,
                                    "bottom_url": session.active_bottom_url
                                }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "STATUS_ACK",
                                    "action": "SET_CLOTH",
                                    "status": "failed"
                                }))
                        else:
                            # Clear all if URL is null
                            session.active_top = None
                            session.active_bottom = None
                            session.active_top_url = None
                            session.active_bottom_url = None
                            await websocket.send_text(json.dumps({
                                "type": "STATUS_ACK",
                                "action": "SET_CLOTH",
                                "status": "cleared"
                            }))

                    elif event_type == "CLEAR_SLOT":
                        slot = payload.get("slot", "ALL").upper()
                        if slot in ("TOP", "ALL"):
                            session.active_top = None
                            session.active_top_url = None
                        if slot in ("BOTTOM", "ALL"):
                            session.active_bottom = None
                            session.active_bottom_url = None
                        await websocket.send_text(json.dumps({
                            "type": "STATUS_ACK",
                            "action": "CLEAR_SLOT",
                            "slot": slot,
                            "has_top": session.active_top is not None,
                            "has_bottom": session.active_bottom is not None
                        }))

                    elif event_type == "SET_MODE":
                        mode = payload.get("mode", "AUTO").upper()
                        session.category_mode = mode
                        logger.info(f"Category selection mode set to: {mode}")

                    elif event_type == "SET_LIGHTING_INTENSITY":
                        val = float(payload.get("intensity", 0.85))
                        session.lighting_intensity = np.clip(val, 0.0, 1.0)
                        logger.info(f"Lighting intensity updated to: {session.lighting_intensity:.2f}")

                    elif event_type == "TOGGLE_MODE":
                        mode = payload.get("mode", "ai")
                        session.view_mode = mode

                except json.JSONDecodeError:
                    logger.warning(f"Malformed JSON: {raw_text}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_ip}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        stop_event.set()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    logger.info("Starting Anywear VTO Step 3 Server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
