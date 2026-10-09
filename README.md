# Anywear VTO - Live Virtual Try-On Chrome Extension & Computer Vision System

[![Manifest V3](https://img.shields.io/badge/Chrome_Extension-Manifest_V3-4285F4.svg)](https://developer.chrome.com/docs/extensions/mv3/intro/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.143-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/Computer_Vision-MediaPipe_Pose-FF6F00.svg)](https://developers.google.com/mediapipe)
[![Throughput](https://img.shields.io/badge/Throughput-35+_FPS-brightgreen.svg)](#benchmark-results)

An enterprise-grade, ultra-low-latency, live webcam Virtual Try-On Chrome Extension (Manifest V3) and Computer Vision backend inspired by **Anywear VTO**.

---

## 🌟 Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               CHROME BROWSER (ANY SHOPPING TAB)                         │
│                                                                                         │
│  ┌───────────────────────┐            ┌──────────────────────────────────────────────┐  │
│  │ Host E-Commerce Store │            │ Anywear VTO Shadow DOM Modal                 │  │
│  │ (Zara / Amazon / etc) │            │                                              │  │
│  │                       │ Right-Click│ ┌──────────────────────────────────────────┐ │  │
│  │ [ Garment Image ] ────┼───────────►│ │ Dual Garment Slots                       │ │  │
│  │   (Tops / Bottoms)    │  or Picker │ │  • TOP:    [ Biker Jacket Thumbnail ]    │ │  │
│  └───────────────────────┘            │ │  • BOTTOM: [ Slim Jeans Thumbnail ]      │ │  │
│                                       │ └────────────────────┬─────────────────────┘ │  │
│  ┌───────────────────────┐            │                      │ SET_CLOTH             │  │
│  │ navigator.mediaDevices│            │ ┌────────────────┐ ┌─▼────────────────────┐  │  │
│  │ .getUserMedia()       │───────────►│ │ WebCam Stream  │ │ AI Real-Time Canvas  │  │  │
│  │ Zero-lag 640x480      │            │ │ <video>        │ │ <canvas> (Feathered) │  │  │
│  └───────────────────────┘            │ └────────┬───────┘ └──────────▲───────────┘  │  │
│                                       │          │                    │              │  │
│  ┌───────────────────────┐            │ ┌────────▼────────────────────┴────────────┐ │  │
│  │ chrome.storage.local  │◄───────────┼─┤ Preferences Persistence (Slider, Mirror) ├─┼──┤
│  └───────────────────────┘            │ └──────────────────────────────────────────┘ │  │
│                                       └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┬──────────────────────────────┘
                                         Binary JPEG Frame │ ~35 FPS Processed
                                             (18 - 20 FPS) │ Binary JPEG Stream
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION CV STREAMING SERVER (server/app.py)                       │
│                                                                                         │
│  1. Ingestion: Non-blocking asynchronous WebSocket receiver (`/ws/stream`)              │
│  2. Multithreaded Worker Pool:                                                          │
│     ├── pose_detector.py: MediaPipe PoseLandmarker + Adaptive EMA Landmark Smoothing    │
│     ├── body_parser.py: Upper Torso vs. Lower Torso Isolation (Preserves Skin & Face)   │
│     ├── depth_shading.py: Real-world Fabric Wrinkle Separation + 3D Diffuse Normals     │
│     └── vton_engine.py: Piecewise Affine Warping + Micro-Draping Spring Physics         │
│  3. JPEG Re-encoder: High-throughput binary compression to browser overlay              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Features at a Glance

1. **Manifest V3 Scoped & Hardened**:
   - Zero host CSS pollution using isolated **Shadow DOM** (`#anywear-vton-root`).
   - Permissions restricted to `activeTab`, `scripting`, `storage`, and `contextMenus`.
   - Auto-injection on existing tabs upon installation/update.

2. **Dual Garment Slots (Simultaneous or Independent Swapping)**:
   - Try on **Tops** (Shirts, Jackets, Tees) and **Bottoms** (Pants, Jeans, Shorts) together or individually.
   - Selecting pants only replaces the lower body while keeping your current shirt 100% visible!

3. **Photorealistic Depth Shading & Wrinkle Synthesis**:
   - Extracts real-world folds and fabric wrinkles from your physical shirt via frequency separation and transfers them onto synthetic garments using Pegtop Soft-Light blending.
   - Computes 3D cylindrical surface normals and anatomical contact ambient occlusion (armpits, collar, waist).

4. **Micro-Draping Spring Physics**:
   - 2D elastic cloth inertia ($a = k \cdot \Delta x - c \cdot v$) gives hemlines natural physical sway as you shift side-to-side.

5. **Right-Click Context Menu Integration**:
   - Right-click any garment photo on any shopping website $\rightarrow$ click **"Try on this garment with Anywear VTO"** to instantly test it.

6. **Instant Snapshot / Photo Capture**:
   - Click the camera button in the modal to immediately download a branded PNG snapshot of your virtual try-on look.

7. **Preferences Persistence (`chrome.storage.local`)**:
   - Remembers your modal drag position, lighting intensity slider value, camera mirror state, and active category mode across browser sessions.

---

## 🚀 Quickstart (Single-Click Windows Launch)

Double-click the bundled launcher in the workspace root:

```powershell
run_system.bat
```

*This will automatically check Python, verify dependencies, execute the health check diagnostic, and launch the server at `http://localhost:8000`.*

---

## 🛠️ Manual CLI Launch & Verification

### Step 1: Run the Automated System Health Check

```powershell
cd "d:\My Projects\anywear-vton-clone\server"
python health_check.py
```

Expected Output:
```
======================================================================
       ANYWEAR VTO - SYSTEM DIAGNOSTIC & HEALTH CHECK
======================================================================

1. Python Runtime:
  [PASS] Detected Python 3.11.9 (Minimum required: 3.9)

2. Core Dependencies:
  [PASS] OpenCV Computer Vision Engine (cv2)
  [PASS] NumPy Matrix & Vector Acceleration (numpy)
  [PASS] SciPy Scientific Computing (scipy)
  [PASS] FastAPI Asynchronous Framework (fastapi)
  [PASS] Uvicorn ASGI High-Performance Server (uvicorn)
  [PASS] Low-Latency WebSocket Transport (websockets)
  [PASS] HTTP Product Image Fetcher (requests)
  [PASS] MediaPipe PoseLandmarker Tasks API (mediapipe)

3. Computer Vision Model Assets:
  [PASS] pose_landmarker_lite.task (5.51 MB)

4. Network Socket Availability:
  [PASS] Port 8000 is open and ready for WebSocket streaming

5. Camera Hardware Probe:
  [PASS] Default Camera Detected (Index 0: 640x480)

======================================================================
  HEALTH CHECK RESULT: [HEALTHY & READY FOR PRODUCTION]
  Launch Command: python server\app.py
======================================================================
```

---

### Step 2: Run the Performance & Realism Benchmark

```powershell
cd "d:\My Projects\anywear-vton-clone\server"
python test_realism.py
```

Benchmark Throughput:
```
  -> Average Render Time: 28.21 ms
  -> Throughput: 35.4 FPS (Target: 25+ FPS)
  -> Torso pixel difference after bottom swap: 0.000 (Selective isolation verified!)
```

---

### Step 3: Launch the Production Server

```powershell
cd "d:\My Projects\anywear-vton-clone\server"
python app.py
```

---

### Step 4: Load into Google Chrome

1. Open **Google Chrome** $\rightarrow$ navigate to `chrome://extensions/`.
2. Toggle **Developer mode** to **ON** (top-right corner).
3. Click **"Load unpacked"** (top-left corner).
4. Select the project folder:
   ```
   d:\My Projects\anywear-vton-clone
   ```
5. Open [`test_store.html`](file:///d:/My%20Projects/anywear-vton-clone/test_store.html) in Chrome.
6. Grant webcam access.
7. Click the toolbar icon or use the **"Pick Garment"** button / **Right-Click Context Menu** to try on any piece!

---

## 📦 Chrome Web Store Packaging Guide

To package this extension for the Chrome Web Store:

1. **Create the Extension ZIP Archive**:
   Zip only the extension client files (exclude the Python server folder and virtual environments):
   ```
   anywear-vton-extension.zip
   ├── manifest.json
   ├── background.js
   ├── content.js
   ├── content.css
   └── icons/
       ├── icon16.png
       ├── icon48.png
       └── icon128.png
   ```

2. **Publish on the Chrome Developer Dashboard**:
   - Navigate to [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole/).
   - Click **Add new item** and upload `anywear-vton-extension.zip`.
   - Fill in Store Listing Details:
     - **Title**: Anywear VTO - Live Virtual Try-On
     - **Category**: Shopping / Productivity
     - **Permissions Justification**:
       - `activeTab`: Inject overlay on user command.
       - `storage`: Persist user lighting & layout preferences.
       - `contextMenus`: Allow right-clicking garment photos to try on.
   - Upload promotional screenshots and submit for review.
