/**
 * Anywear VTO - Live Virtual Try-On Content Script (Manifest V3)
 * Provides:
 *  1. Glassmorphic, draggable & minimizable floating UI modal (isolated in Shadow DOM).
 *  2. Zero-lag local webcam streaming via getUserMedia.
 *  3. Global e-commerce garment image picker with high-res resolution heuristics.
 *  4. High-performance WebSocket binary frame streaming to Python FastAPI CV backend.
 *  5. Low-latency server-rendered canvas overlay with real-time FPS & RTT latency metrics.
 */

(function () {
  // Prevent duplicate script execution
  if (window.__ANYWEAR_VTO_INITIALIZED__) {
    console.log("[Anywear VTO] Content script already initialized in this frame.");
    return;
  }
  window.__ANYWEAR_VTO_INITIALIZED__ = true;

  // =========================================================================
  // Domain Exclusion & Privacy Guards
  // =========================================================================
  const EXCLUDED_HOSTS = [
    "gemini.google.com",
    "github.com",
    "google.com",
    "www.google.com",
    "mail.google.com",
    "drive.google.com",
    "docs.google.com",
    "stackoverflow.com",
    "chatgpt.com",
    "claude.ai"
  ];

  function isExcludedHost() {
    try {
      const host = (window.location.hostname || "").toLowerCase();
      const protocol = window.location.protocol;
      if (protocol === "chrome:" || protocol === "edge:" || protocol === "about:" || protocol === "chrome-extension:") {
        return true;
      }
      return EXCLUDED_HOSTS.some(excluded => host === excluded || host.endsWith("." + excluded));
    } catch {
      return false;
    }
  }

  // =========================================================================
  // State Management
  // =========================================================================
  const state = {
    isOpen: true,
    isMinimized: false,
    isCameraRunning: false,
    isMirrored: true,
    isPickingGarment: false,
    viewMode: "ai", // "ai" (server rendered) | "camera" (raw webcam)
    activeClothUrl: null,
    activeTopUrl: null,
    activeBottomUrl: null,
    categoryMode: "AUTO", // "AUTO" | "TOP" | "BOTTOM"
    lightingIntensity: 0.85,
    
    // WebSocket & Streaming
    wsUrl: "ws://localhost:8000/ws/stream",
    ws: null,
    wsConnected: false,
    reconnectTimer: null,
    reconnectAttempts: 0,
    maxReconnectDelay: 8000,
    
    // Video streaming parameters (Ultra-low latency ping-pong lock)
    targetFps: 30,
    frameIntervalMs: 33,
    lastFrameSentAt: 0,
    isFrameInFlight: false,
    frameInFlightTime: 0,
    pendingFrameCount: 0,
    streamTimer: null,
    
    // Performance & Latency Telemetry
    sentFrameTimestamps: new Map(), // frameSeq -> clientTimestamp
    frameSeq: 0,
    fpsCalc: {
      clientFrames: 0,
      serverFrames: 0,
      lastCalculated: Date.now(),
      displayClientFps: 0,
      displayServerFps: 0,
      currentLatencyMs: 0
    }
  };

  // Local references to media streams and elements
  let mediaStream = null;
  let captureCanvas = null;
  let captureCtx = null;
  let shadowRoot = null;
  let rootContainer = null;
  let elements = {};

  // =========================================================================
  // 1. Host Page Picker Elements (Outside Shadow DOM for Global Highlight)
  // =========================================================================
  let pickerBox = null;
  let pickerBadge = null;

  function initHostPickerOverlay() {
    pickerBox = document.createElement("div");
    pickerBox.id = "anywear-picker-box";
    pickerBadge = document.createElement("div");
    pickerBadge.id = "anywear-picker-badge";
    pickerBadge.textContent = "Select Garment";
    pickerBox.appendChild(pickerBadge);
    document.body.appendChild(pickerBox);

    // Global event listeners for garment hover & click
    document.addEventListener("mouseover", handleHostPageHover, true);
    document.addEventListener("click", handleHostPageClick, true);
    document.addEventListener("keydown", handleHostPageKeyDown, true);
  }

  function toggleGarmentPicker(forceState) {
    state.isPickingGarment = typeof forceState === "boolean" ? forceState : !state.isPickingGarment;
    
    if (state.isPickingGarment) {
      document.body.classList.add("anywear-picker-active");
      if (elements.pickClothBtn) {
        elements.pickClothBtn.classList.add("picking");
        elements.pickClothBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          Cancel Pick
        `;
      }
    } else {
      document.body.classList.remove("anywear-picker-active");
      if (pickerBox) pickerBox.style.display = "none";
      if (elements.pickClothBtn) {
        elements.pickClothBtn.classList.remove("picking");
        elements.pickClothBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v8M8 12h8"></path></svg>
          Pick Garment
        `;
      }
    }
  }

  function handleHostPageKeyDown(e) {
    if (state.isPickingGarment && e.key === "Escape") {
      toggleGarmentPicker(false);
    }
  }

  function findGarmentImageElement(target) {
    if (!target) return null;
    // Disallow picking anything inside our extension modal
    if (rootContainer && rootContainer.contains(target)) return null;
    if (target === pickerBox || pickerBox.contains(target)) return null;

    // Check if target itself is an image
    if (target.tagName === "IMG") return target;

    // Check closest picture or container with img
    const imgInside = target.querySelector("img");
    if (imgInside) return imgInside;

    const parentImg = target.closest("picture")?.querySelector("img");
    if (parentImg) return parentImg;

    // Check if element has background-image style
    const bg = window.getComputedStyle(target).backgroundImage;
    if (bg && bg !== "none" && bg.startsWith("url(")) {
      return target;
    }

    return null;
  }

  function handleHostPageHover(e) {
    if (!state.isPickingGarment) return;

    const imgEl = findGarmentImageElement(e.target);
    if (!imgEl) {
      if (pickerBox) pickerBox.style.display = "none";
      return;
    }

    const rect = imgEl.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) return; // ignore tiny icons

    const scrollX = window.scrollX || window.pageXOffset;
    const scrollY = window.scrollY || window.pageYOffset;

    pickerBox.style.display = "block";
    pickerBox.style.left = `${rect.left + scrollX}px`;
    pickerBox.style.top = `${rect.top + scrollY}px`;
    pickerBox.style.width = `${rect.width}px`;
    pickerBox.style.height = `${rect.height}px`;
  }

  /**
   * Advanced High-Res Image Extractor:
   * Examines srcset, high-res data attributes common in Shopify, Amazon, Zara, Magento, etc.
   */
  function extractHighResImageUrl(el) {
    if (!el) return null;

    // 1. Check data attributes first (commonly store 1500px zoom images)
    const highResAttrs = [
      "data-zoom-image",
      "data-large-img",
      "data-zoom",
      "data-high-res-src",
      "data-original",
      "data-full",
      "data-large",
      "data-src",
      "data-lazy-src",
      "data-old-hires"
    ];

    for (const attr of highResAttrs) {
      const val = el.getAttribute(attr);
      if (val && val.trim().length > 0) {
        return resolveUrl(val.trim());
      }
    }

    // 2. Parse srcset if available
    const srcset = el.getAttribute("srcset");
    if (srcset) {
      const candidates = srcset.split(",").map(entry => {
        const parts = entry.trim().split(/\s+/);
        const url = parts[0];
        let width = 0;
        if (parts[1]) {
          if (parts[1].endsWith("w")) width = parseInt(parts[1], 10);
          else if (parts[1].endsWith("x")) width = parseFloat(parts[1]) * 1000;
        }
        return { url, width };
      });

      candidates.sort((a, b) => b.width - a.width);
      if (candidates.length > 0 && candidates[0].url) {
        return resolveUrl(candidates[0].url);
      }
    }

    // 3. Fallback to standard src or currentSrc
    if (el.currentSrc) return resolveUrl(el.currentSrc);
    if (el.src) return resolveUrl(el.src);

    // 4. Background-image extraction
    const bg = window.getComputedStyle(el).backgroundImage;
    if (bg && bg !== "none" && bg.startsWith("url(")) {
      const cleanBg = bg.replace(/^url\(["']?/, "").replace(/["']?\)$/, "");
      return resolveUrl(cleanBg);
    }

    return null;
  }

  function resolveUrl(urlStr) {
    try {
      return new URL(urlStr, window.location.href).href;
    } catch {
      return urlStr;
    }
  }

  function handleHostPageClick(e) {
    if (!state.isPickingGarment) return;

    const imgEl = findGarmentImageElement(e.target);
    if (!imgEl) return;

    e.preventDefault();
    e.stopPropagation();

    const highResUrl = extractHighResImageUrl(imgEl);
    if (highResUrl) {
      console.log("[Anywear VTO] Selected Garment URL:", highResUrl);
      setGarment(highResUrl, imgEl.alt || "Selected Garment");
    }

    toggleGarmentPicker(false);
  }

  function setGarment(url, title = "Garment Item") {
    state.activeClothUrl = url;

    // Send SET_CLOTH metadata event to backend over WebSocket with active slot mode
    sendWebSocketJson({
      type: "SET_CLOTH",
      cloth_url: url,
      slot: state.categoryMode,
      title: title,
      timestamp: Date.now()
    });
  }

  function clearSlot(slot) {
    if (slot === "TOP") {
      state.activeTopUrl = null;
      updateSlotUi("TOP", null, "No Top active", "Pick shirt or jacket");
    } else if (slot === "BOTTOM") {
      state.activeBottomUrl = null;
      updateSlotUi("BOTTOM", null, "No Bottom active", "Pick jeans or pants");
    } else {
      state.activeTopUrl = null;
      state.activeBottomUrl = null;
      updateSlotUi("TOP", null, "No Top active", "Pick shirt or jacket");
      updateSlotUi("BOTTOM", null, "No Bottom active", "Pick jeans or pants");
    }

    sendWebSocketJson({
      type: "CLEAR_SLOT",
      slot: slot,
      timestamp: Date.now()
    });
  }

  function updateSlotUi(slot, url, title, subtitle) {
    if (slot === "TOP") {
      if (elements.topThumb) {
        elements.topThumb.innerHTML = url
          ? `<img src="${url}" alt="Top Preview" />`
          : `<svg class="vto-garment-thumb-placeholder" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20.38 3.46L16 2a4 4 0 01-8 0L3.62 3.46a2 2 0 00-1.34 2.23l.58 3.47a1 1 0 00.99.84H6v10c0 1.1.9 2 2 2h8a2 2 0 002-2V10h2.15a1 1 0 00.99-.84l.58-3.47a2 2 0 00-1.34-2.23z"/></svg>`;
      }
      if (elements.topTitle) elements.topTitle.textContent = title;
      if (elements.topSub) elements.topSub.textContent = subtitle;
      if (elements.topClearBtn) elements.topClearBtn.style.display = url ? "block" : "none";
    } else if (slot === "BOTTOM") {
      if (elements.bottomThumb) {
        elements.bottomThumb.innerHTML = url
          ? `<img src="${url}" alt="Bottom Preview" />`
          : `<svg class="vto-garment-thumb-placeholder" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 3h12l1 7-2 11h-4l-1-9-1 9H7L5 10l1-7z"/></svg>`;
      }
      if (elements.bottomTitle) elements.bottomTitle.textContent = title;
      if (elements.bottomSub) elements.bottomSub.textContent = subtitle;
      if (elements.bottomClearBtn) elements.bottomClearBtn.style.display = url ? "block" : "none";
    }
  }

  function setCategoryMode(mode) {
    state.categoryMode = mode;
    if (elements.catAutoBtn) elements.catAutoBtn.classList.toggle("active", mode === "AUTO");
    if (elements.catTopBtn) elements.catTopBtn.classList.toggle("active", mode === "TOP");
    if (elements.catBottomBtn) elements.catBottomBtn.classList.toggle("active", mode === "BOTTOM");

    sendWebSocketJson({
      type: "SET_MODE",
      mode: mode,
      timestamp: Date.now()
    });
  }

  function setLightingIntensity(val) {
    state.lightingIntensity = val;
    if (elements.sliderVal) {
      elements.sliderVal.textContent = `${Math.round(val * 100)}%`;
    }
    sendWebSocketJson({
      type: "SET_LIGHTING_INTENSITY",
      intensity: val,
      timestamp: Date.now()
    });
  }

  // =========================================================================
  // 2. Persistent WebSocket Connection to Python FastAPI Backend
  // =========================================================================
  function initWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    updateConnectionUi("connecting");
    console.log(`[Anywear VTO] Connecting to backend at ${state.wsUrl}...`);

    try {
      state.ws = new WebSocket(state.wsUrl);
      state.ws.binaryType = "arraybuffer"; // High-efficiency binary transport

      state.ws.onopen = () => {
        console.log("[Anywear VTO] WebSocket connected successfully!");
        state.wsConnected = true;
        state.reconnectAttempts = 0;
        updateConnectionUi("connected");

        // Sync initial state with server
        sendWebSocketJson({
          type: "CLIENT_INIT",
          mode: state.viewMode,
          cloth_url: state.activeClothUrl,
          timestamp: Date.now()
        });

        // Start video frame streaming loop if webcam is active
        startFrameStreamLoop();
      };

      state.ws.onmessage = async (event) => {
        state.fpsCalc.serverFrames++;

        if (event.data instanceof ArrayBuffer) {
          // Binary JPEG frame returned by backend
          handleServerBinaryFrame(event.data);
        } else if (typeof event.data === "string") {
          // JSON metadata event
          try {
            const data = JSON.parse(event.data);
            handleServerJsonMessage(data);
          } catch (e) {
            console.warn("[Anywear VTO] Received non-JSON string message:", event.data);
          }
        }
      };

      state.ws.onclose = (event) => {
        console.warn(`[Anywear VTO] WebSocket closed (code: ${event.code}). Scheduling reconnect...`);
        state.wsConnected = false;
        state.isFrameInFlight = false;
        state.sentFrameTimestamps.clear();
        updateConnectionUi("disconnected");
        scheduleReconnect();
      };

      state.ws.onerror = (err) => {
        console.error("[Anywear VTO] WebSocket error:", err);
        state.wsConnected = false;
        state.isFrameInFlight = false;
        state.sentFrameTimestamps.clear();
        updateConnectionUi("error");
      };
    } catch (e) {
      console.error("[Anywear VTO] Failed to instantiate WebSocket:", e);
      scheduleReconnect();
    }
  }

  function scheduleReconnect() {
    if (state.reconnectTimer) clearTimeout(state.reconnectTimer);
    
    // Exponential backoff capped at maxReconnectDelay
    const delay = Math.min(1500 * Math.pow(1.5, state.reconnectAttempts), state.maxReconnectDelay);
    state.reconnectAttempts++;

    state.reconnectTimer = setTimeout(() => {
      console.log(`[Anywear VTO] Reconnecting attempt #${state.reconnectAttempts}...`);
      initWebSocket();
    }, delay);
  }

  function updateConnectionUi(status) {
    if (!elements.statusDot || !elements.statusText) return;

    elements.statusDot.className = "vto-status-dot";

    if (status === "connected") {
      elements.statusDot.classList.add("connected");
      elements.statusText.textContent = "AI Live";
      elements.statusText.style.color = "#34d399";
    } else if (status === "connecting") {
      elements.statusDot.classList.add("connecting");
      elements.statusText.textContent = "Connecting...";
      elements.statusText.style.color = "#fbbf24";
    } else {
      elements.statusText.textContent = "Backend Offline";
      elements.statusText.style.color = "#f87171";
    }
  }

  function sendWebSocketJson(payload) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(payload));
    }
  }

  // =========================================================================
  // 3. Webcam Media Stream & Zero-Lag Local Video
  // =========================================================================
  function showCameraPermissionBanner(title = "Click to allow camera", description = "Click below to enable your camera for live try-on.") {
    if (!elements.placeholder) return;
    elements.placeholder.classList.remove("hidden");
    elements.placeholder.innerHTML = `
      <div class="vto-permission-card" style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:18px 14px; gap:8px;">
        <svg class="vto-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="width:36px; height:36px; color:#818cf8; margin-bottom:2px;">
          <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
          <circle cx="12" cy="13" r="4"></circle>
        </svg>
        <div style="font-size:13px; font-weight:600; color:#f8fafc;">${title}</div>
        <div style="font-size:11px; color:#94a3b8; max-width:220px; line-height:1.4;">${description}</div>
        <button id="vto-allow-cam-btn" class="vto-btn vto-btn-primary" style="margin-top:6px; padding:6px 14px; font-size:11px; font-weight:600; border-radius:7px; cursor:pointer;">
          Turn on Camera
        </button>
      </div>
    `;
    const allowBtn = shadowRoot.getElementById("vto-allow-cam-btn");
    if (allowBtn) {
      allowBtn.addEventListener("click", () => {
        startWebcam();
      });
    }
  }

  async function startWebcam() {
    if (isExcludedHost()) {
      console.info("[Anywear VTO] Skipping webcam on excluded domain:", window.location.hostname);
      showCameraPermissionBanner("Camera Standby", "Camera is disabled on this page.");
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn("[Anywear VTO] navigator.mediaDevices.getUserMedia unavailable in this context.");
      showCameraPermissionBanner("Camera Not Supported", "Webcam access is unavailable in this browser frame.");
      return;
    }

    try {
      if (elements.placeholder) elements.placeholder.classList.add("hidden");

      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
        },
        audio: false
      };

      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (elements.video) {
        elements.video.srcObject = mediaStream;
        await elements.video.play();
        state.isCameraRunning = true;

        if (elements.toggleCamBtn) {
          elements.toggleCamBtn.classList.add("active");
          elements.toggleCamBtn.title = "Turn Off Camera";
        }

        // Initialize offscreen capture canvas downscaled to 480x360 for ultra-low latency
        const track = mediaStream.getVideoTracks()[0];
        const settings = track.getSettings();
        const rawWidth = settings.width || 640;
        const rawHeight = settings.height || 480;

        // Clamped downscale to 480x360 (or preserve aspect ratio with max dimension 480)
        const targetWidth = 480;
        const targetHeight = Math.round(targetWidth * (rawHeight / rawWidth)) || 360;

        captureCanvas = document.createElement("canvas");
        captureCanvas.width = targetWidth;
        captureCanvas.height = targetHeight;
        captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

        // Synchronize on-screen overlay canvas resolution
        if (elements.canvas) {
          elements.canvas.width = targetWidth;
          elements.canvas.height = targetHeight;
        }

        state.isFrameInFlight = false;
        state.sentFrameTimestamps.clear();
        startFrameStreamLoop();
      }
    } catch (err) {
      // Graceful fallback without unhandled red console.error
      console.info("[Anywear VTO] Camera access paused or permission required:", err?.name || err?.message || err);
      state.isCameraRunning = false;
      state.isFrameInFlight = false;
      state.sentFrameTimestamps.clear();
      if (elements.toggleCamBtn) {
        elements.toggleCamBtn.classList.remove("active");
        elements.toggleCamBtn.title = "Turn On Camera";
      }
      showCameraPermissionBanner("Click to allow camera", "Camera permission is needed to stream real-time virtual try-on.");
    }
  }

  function stopWebcam() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    if (elements.video) {
      elements.video.srcObject = null;
    }
    state.isCameraRunning = false;
    state.isFrameInFlight = false;
    state.frameInFlightTime = 0;
    state.sentFrameTimestamps.clear();
    if (state.streamTimer) {
      clearInterval(state.streamTimer);
      state.streamTimer = null;
    }
    if (elements.toggleCamBtn) {
      elements.toggleCamBtn.classList.remove("active");
      elements.toggleCamBtn.title = "Turn On Camera";
    }
    showCameraPermissionBanner("Camera Paused", "Click below to turn on the camera.");
  }

  function toggleWebcam() {
    if (state.isCameraRunning) {
      stopWebcam();
    } else {
      startWebcam();
    }
  }

  function toggleMirror() {
    state.isMirrored = !state.isMirrored;
    if (elements.video) {
      if (state.isMirrored) elements.video.classList.add("mirrored");
      else elements.video.classList.remove("mirrored");
    }
    if (elements.mirrorBtn) {
      elements.mirrorBtn.classList.toggle("active", state.isMirrored);
    }
  }

  // =========================================================================
  // 4. Low-Latency Frame Capture & WebSocket Transmission (In-Flight Lock)
  // =========================================================================
  function startFrameStreamLoop() {
    if (state.streamTimer) return;
    state.isFrameInFlight = false;
    state.streamTimer = setInterval(captureAndSendFrame, state.frameIntervalMs);
  }

  function captureAndSendFrame() {
    if (!state.isCameraRunning || !state.wsConnected) return;
    if (!elements.video || elements.video.readyState < 2) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    // IN-FLIGHT FRAME LOCK:
    // NEVER send a new frame if the previous frame has not yet returned from the server.
    if (state.isFrameInFlight) {
      // Watchdog failsafe: if a frame is lost/dropped for >250ms, unlock so stream never stalls
      if (Date.now() - state.frameInFlightTime > 250) {
        state.isFrameInFlight = false;
        state.sentFrameTimestamps.clear();
      } else {
        return; // Skip this tick! Eliminates queue buffer bloat and latency buildup completely.
      }
    }

    // Backpressure safeguard: skip if socket buffer has queued data
    if (state.ws.bufferedAmount > 64 * 1024) {
      return;
    }

    const width = captureCanvas.width;
    const height = captureCanvas.height;

    // Draw current webcam frame to downscaled offscreen canvas
    captureCtx.save();
    if (state.isMirrored) {
      // Mirror horizontally so server processes natural selfie perspective
      captureCtx.translate(width, 0);
      captureCtx.scale(-1, 1);
    }
    captureCtx.drawImage(elements.video, 0, 0, width, height);
    captureCtx.restore();

    state.isFrameInFlight = true;
    state.frameInFlightTime = Date.now();

    // Export frame as JPEG Blob (quality: 0.65 for instant downscaled serialization)
    captureCanvas.toBlob((blob) => {
      if (!blob || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        state.isFrameInFlight = false;
        return;
      }

      const seq = ++state.frameSeq;
      const sendTime = Date.now();
      state.sentFrameTimestamps.set(seq, sendTime);

      // Keep only current sequence to avoid unbounded Map growth
      if (state.sentFrameTimestamps.size > 5) {
        state.sentFrameTimestamps.clear();
        state.sentFrameTimestamps.set(seq, sendTime);
      }

      blob.arrayBuffer().then((buffer) => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
          state.ws.send(buffer);
          state.fpsCalc.clientFrames++;
        } else {
          state.isFrameInFlight = false;
        }
      }).catch(() => {
        state.isFrameInFlight = false;
      });
    }, "image/jpeg", 0.65);
  }

  // =========================================================================
  // 5. Server Frame Rendering & Overlay Canvas Processing
  // =========================================================================
  async function handleServerBinaryFrame(arrayBuffer) {
    // Release in-flight lock immediately
    state.isFrameInFlight = false;

    // Calculate exact Round-Trip Latency (RTT)
    const now = Date.now();
    if (state.frameInFlightTime > 0) {
      state.fpsCalc.currentLatencyMs = Math.max(1, now - state.frameInFlightTime);
    }
    state.sentFrameTimestamps.clear();

    if (!elements.canvas) return;

    try {
      const blob = new Blob([arrayBuffer], { type: "image/jpeg" });
      const imageBitmap = await createImageBitmap(blob);

      const ctx = elements.canvas.getContext("2d");
      ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
      ctx.drawImage(imageBitmap, 0, 0, elements.canvas.width, elements.canvas.height);
    } catch (err) {
      console.error("[Anywear VTO] Error rendering server frame:", err);
    }
  }

  function handleServerJsonMessage(data) {
    if (data.type === "STATUS_ACK") {
      if (typeof data.server_fps === "number" && elements.fpsPill) {
        elements.fpsPill.textContent = `${Math.round(data.server_fps)} FPS`;
      }

      if (data.action === "SET_CLOTH") {
        if (data.status === "ready") {
          if (data.assigned_slot === "TOP") {
            state.activeTopUrl = data.top_url;
            updateSlotUi("TOP", data.top_url, "Top Garment Active", "Folds & Depth Synced");
          } else {
            state.activeBottomUrl = data.bottom_url;
            updateSlotUi("BOTTOM", data.bottom_url, "Bottom Garment Active", "Legs Fitted");
          }
        }
      } else if (data.action === "CLEAR_SLOT") {
        if (data.slot === "TOP" || data.slot === "ALL") {
          updateSlotUi("TOP", null, "No Top active", "Pick shirt or jacket");
        }
        if (data.slot === "BOTTOM" || data.slot === "ALL") {
          updateSlotUi("BOTTOM", null, "No Bottom active", "Pick jeans or pants");
        }
      }
    }
  }

  // Periodic FPS & Latency UI updates (every 1 second)
  setInterval(() => {
    const now = Date.now();
    const elapsedSec = (now - state.fpsCalc.lastCalculated) / 1000;
    
    if (elapsedSec > 0.5) {
      const clientFps = Math.round(state.fpsCalc.clientFrames / elapsedSec);
      const serverFps = Math.round(state.fpsCalc.serverFrames / elapsedSec);

      state.fpsCalc.clientFrames = 0;
      state.fpsCalc.serverFrames = 0;
      state.fpsCalc.lastCalculated = now;

      if (elements.fpsPill) {
        elements.fpsPill.textContent = `${serverFps > 0 ? serverFps : clientFps} FPS`;
      }
      if (elements.latencyPill && state.fpsCalc.currentLatencyMs > 0) {
        elements.latencyPill.textContent = `${state.fpsCalc.currentLatencyMs}ms`;
      }
    }
  }, 1000);

  // =========================================================================
  // 6. View Mode Toggling (AI Canvas vs Raw Video)
  // =========================================================================
  function setViewMode(mode) {
    state.viewMode = mode;

    if (elements.modeAiBtn && elements.modeCamBtn) {
      elements.modeAiBtn.classList.toggle("active", mode === "ai");
      if (elements.modeSkelBtn) elements.modeSkelBtn.classList.toggle("active", mode === "skeleton");
      elements.modeCamBtn.classList.toggle("active", mode === "camera");
    }

    if (elements.canvas) {
      if (mode === "camera") {
        elements.canvas.classList.add("hidden");
      } else {
        elements.canvas.classList.remove("hidden");
      }
    }

    sendWebSocketJson({
      type: "TOGGLE_MODE",
      mode: mode,
      timestamp: Date.now()
    });
  }

  // =========================================================================
  // 7. Draggable Floating Modal Window Logic
  // =========================================================================
  function makeElementDraggable(dragHandle, targetWindow) {
    let isDragging = false;
    let startX = 0, startY = 0;
    let initialLeft = 0, initialTop = 0;

    dragHandle.addEventListener("mousedown", (e) => {
      // Don't drag if clicking buttons inside the header
      if (e.target.closest("button")) return;

      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;

      const rect = targetWindow.getBoundingClientRect();
      initialLeft = rect.left;
      initialTop = rect.top;

      // Set explicit left/top and remove right/bottom positioning
      targetWindow.style.left = `${initialLeft}px`;
      targetWindow.style.top = `${initialTop}px`;
      targetWindow.style.right = "auto";
      targetWindow.style.bottom = "auto";

      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
      e.preventDefault();
    });

    function onMouseMove(e) {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;

      let newLeft = initialLeft + dx;
      let newTop = initialTop + dy;

      // Restrict within viewport boundaries
      const maxLeft = window.innerWidth - targetWindow.offsetWidth - 10;
      const maxTop = window.innerHeight - targetWindow.offsetHeight - 10;

      newLeft = Math.max(10, Math.min(newLeft, maxLeft));
      newTop = Math.max(10, Math.min(newTop, maxTop));

      targetWindow.style.left = `${newLeft}px`;
      targetWindow.style.top = `${newTop}px`;
    }

    function onMouseUp() {
      isDragging = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    }
  }

  function toggleMinimize() {
    state.isMinimized = !state.isMinimized;
    if (elements.window) {
      elements.window.classList.toggle("vto-minimized", state.isMinimized);
    }
    if (elements.minimizeBtn) {
      elements.minimizeBtn.innerHTML = state.isMinimized
        ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>`
        : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line></svg>`;
      elements.minimizeBtn.title = state.isMinimized ? "Expand Modal" : "Minimize Modal";
    }
  }

  function toggleModalVisibility(force) {
    state.isOpen = typeof force === "boolean" ? force : !state.isOpen;
    if (elements.window) {
      elements.window.style.display = state.isOpen ? "flex" : "none";
    }
    if (state.isOpen && !state.isCameraRunning && !isExcludedHost()) {
      startWebcam();
    }
  }

  // =========================================================================
  // 8. Shadow DOM Construction & HTML Template
  // =========================================================================
  function buildModalDom() {
    rootContainer = document.createElement("div");
    rootContainer.id = "anywear-vton-root";
    document.documentElement.appendChild(rootContainer);

    shadowRoot = rootContainer.attachShadow({ mode: "open" });

    // Link Scoped CSS inside Shadow Root
    const linkStyle = document.createElement("link");
    linkStyle.rel = "stylesheet";
    linkStyle.href = chrome.runtime.getURL("content.css");
    shadowRoot.appendChild(linkStyle);

    // Modal Window Template
    const modalMarkup = `
      <div id="vto-window" class="vto-window" style="top: 24px; right: 24px;">
        <!-- Draggable Header -->
        <div id="vto-header" class="vto-header">
          <div class="vto-title-group">
            <div class="vto-logo-icon">
              <svg viewBox="0 0 24 24">
                <path d="M12 2a4 4 0 0 0-4 4v1.17a6 6 0 0 0-5 5.83v7a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7a6 6 0 0 0-5-5.83V6a4 4 0 0 0-4-4zm0 2a2 2 0 0 1 2 2v1h-4V6a2 2 0 0 1 2-2z"/>
              </svg>
            </div>
            <span class="vto-title">ANYWEAR VTO</span>
            <span class="vto-badge">LIVE AI</span>
          </div>
          <div class="vto-window-controls">
            <button id="vto-minimize-btn" class="vto-icon-btn" title="Minimize Modal">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
            </button>
            <button id="vto-close-btn" class="vto-icon-btn close-btn" title="Close Window">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
          </div>
        </div>

        <!-- Body Area -->
        <div class="vto-body">
          <!-- Video & Canvas Viewport -->
          <div class="vto-viewport">
            <video id="vto-video" class="vto-video-element mirrored" autoplay playsinline muted></video>
            <canvas id="vto-canvas" class="vto-canvas-element"></canvas>

            <!-- Video HUD Telemetry -->
            <div class="vto-hud">
              <div class="vto-stat-pill">
                <span id="vto-status-dot" class="vto-status-dot"></span>
                <span id="vto-status-text">Connecting...</span>
              </div>
              <div style="display:flex; gap:6px;">
                <div id="vto-latency-pill" class="vto-stat-pill">-- ms</div>
                <div id="vto-fps-pill" class="vto-stat-pill">-- FPS</div>
              </div>
            </div>

            <!-- View Mode Switcher -->
            <div class="vto-mode-pill">
              <button id="vto-mode-ai" class="vto-mode-btn active">AI Try-On</button>
              <button id="vto-mode-skel" class="vto-mode-btn">Pose Mesh</button>
              <button id="vto-mode-cam" class="vto-mode-btn">Camera</button>
            </div>

            <!-- Camera Permission Fallback -->
            <div id="vto-placeholder" class="vto-viewport-placeholder">
              <div class="vto-permission-card" style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:18px 14px; gap:8px;">
                <svg class="vto-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="width:36px; height:36px; color:#818cf8; margin-bottom:2px;">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                  <circle cx="12" cy="13" r="4"></circle>
                </svg>
                <div style="font-size:13px; font-weight:600; color:#f8fafc;">Click to allow camera</div>
                <div style="font-size:11px; color:#94a3b8; max-width:220px; line-height:1.4;">Click below to enable your camera for live try-on.</div>
                <button id="vto-allow-cam-btn" class="vto-btn vto-btn-primary" style="margin-top:6px; padding:6px 14px; font-size:11px; font-weight:600; border-radius:7px; cursor:pointer;">
                  Turn on Camera
                </button>
              </div>
            </div>
          </div>

          <!-- Category Selector & Lighting Realism Slider -->
          <div class="vto-selector-row">
            <span class="vto-row-label">Category Mode:</span>
            <div class="vto-category-pill">
              <button id="vto-cat-auto" class="vto-cat-btn active">Auto</button>
              <button id="vto-cat-top" class="vto-cat-btn">Top</button>
              <button id="vto-cat-bottom" class="vto-cat-btn">Bottom</button>
            </div>
          </div>

          <div class="vto-slider-row">
            <div class="vto-slider-header">
              <span class="vto-row-label">Fold & Depth Shading</span>
              <span id="vto-slider-val" class="vto-slider-val">85%</span>
            </div>
            <input type="range" id="vto-lighting-slider" class="vto-range-slider" min="0" max="100" value="85" />
          </div>

          <!-- Dual Garment Slots (Top & Bottom) -->
          <div class="vto-slots-container">
            <!-- Top Slot -->
            <div class="vto-slot-card" id="vto-slot-top">
              <div class="vto-slot-badge top-badge">TOP</div>
              <div id="vto-top-thumb" class="vto-slot-thumb">
                <svg class="vto-garment-thumb-placeholder" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                  <path d="M20.38 3.46L16 2a4 4 0 01-8 0L3.62 3.46a2 2 0 00-1.34 2.23l.58 3.47a1 1 0 00.99.84H6v10c0 1.1.9 2 2 2h8a2 2 0 002-2V10h2.15a1 1 0 00.99-.84l.58-3.47a2 2 0 00-1.34-2.23z"/>
                </svg>
              </div>
              <div class="vto-slot-info">
                <div id="vto-top-title" class="vto-slot-title">No Top active</div>
                <div id="vto-top-sub" class="vto-slot-sub">Pick shirt or jacket</div>
              </div>
              <button id="vto-top-clear" class="vto-slot-clear" title="Clear Top" style="display:none;">✕</button>
            </div>

            <!-- Bottom Slot -->
            <div class="vto-slot-card" id="vto-slot-bottom">
              <div class="vto-slot-badge bot-badge">BOTTOM</div>
              <div id="vto-bottom-thumb" class="vto-slot-thumb">
                <svg class="vto-garment-thumb-placeholder" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                  <path d="M6 3h12l1 7-2 11h-4l-1-9-1 9H7L5 10l1-7z"/>
                </svg>
              </div>
              <div class="vto-slot-info">
                <div id="vto-bottom-title" class="vto-slot-title">No Bottom active</div>
                <div id="vto-bottom-sub" class="vto-slot-sub">Pick jeans or pants</div>
              </div>
              <button id="vto-bottom-clear" class="vto-slot-clear" title="Clear Bottom" style="display:none;">✕</button>
            </div>
          </div>

          <!-- Actions -->
          <div class="vto-actions">
            <button id="vto-pick-cloth-btn" class="vto-btn vto-btn-primary">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v8M8 12h8"></path></svg>
              Pick Garment
            </button>
            <button id="vto-reconnect-btn" class="vto-btn vto-btn-secondary" title="Reconnect WebSocket">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
              Reconnect
            </button>
          </div>
        </div>

        <!-- Footer Bar with Media Controls -->
        <div class="vto-footer">
          <div class="vto-footer-left">
            <span>Stream: 480x360 @ 30fps (Low-Latency)</span>
          </div>
          <div class="vto-footer-right">
            <button id="vto-snapshot-btn" class="vto-tool-icon-btn" title="Capture Try-On Photo">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
            </button>
            <button id="vto-toggle-cam" class="vto-tool-icon-btn" title="Turn On Camera">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"></path><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
            </button>
            <button id="vto-toggle-mirror" class="vto-tool-icon-btn active" title="Mirror Camera">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 16V4m0 0L3 8m4-4l4 4M17 8v12m0 0l4-4m-4 4l-4-4"/></svg>
            </button>
          </div>
        </div>

        <!-- Toast Feedback Notification -->
        <div id="vto-toast" class="vto-toast"></div>
      </div>
    `;

    const template = document.createElement("template");
    template.innerHTML = modalMarkup;
    shadowRoot.appendChild(template.content.cloneNode(true));

    // Cache elements
    elements = {
      window: shadowRoot.getElementById("vto-window"),
      header: shadowRoot.getElementById("vto-header"),
      video: shadowRoot.getElementById("vto-video"),
      canvas: shadowRoot.getElementById("vto-canvas"),
      placeholder: shadowRoot.getElementById("vto-placeholder"),
      statusDot: shadowRoot.getElementById("vto-status-dot"),
      statusText: shadowRoot.getElementById("vto-status-text"),
      latencyPill: shadowRoot.getElementById("vto-latency-pill"),
      fpsPill: shadowRoot.getElementById("vto-fps-pill"),
      modeAiBtn: shadowRoot.getElementById("vto-mode-ai"),
      modeSkelBtn: shadowRoot.getElementById("vto-mode-skel"),
      modeCamBtn: shadowRoot.getElementById("vto-mode-cam"),
      catAutoBtn: shadowRoot.getElementById("vto-cat-auto"),
      catTopBtn: shadowRoot.getElementById("vto-cat-top"),
      catBottomBtn: shadowRoot.getElementById("vto-cat-bottom"),
      lightingSlider: shadowRoot.getElementById("vto-lighting-slider"),
      sliderVal: shadowRoot.getElementById("vto-slider-val"),
      topThumb: shadowRoot.getElementById("vto-top-thumb"),
      topTitle: shadowRoot.getElementById("vto-top-title"),
      topSub: shadowRoot.getElementById("vto-top-sub"),
      topClearBtn: shadowRoot.getElementById("vto-top-clear"),
      bottomThumb: shadowRoot.getElementById("vto-bottom-thumb"),
      bottomTitle: shadowRoot.getElementById("vto-bottom-title"),
      bottomSub: shadowRoot.getElementById("vto-bottom-sub"),
      bottomClearBtn: shadowRoot.getElementById("vto-bottom-clear"),
      pickClothBtn: shadowRoot.getElementById("vto-pick-cloth-btn"),
      reconnectBtn: shadowRoot.getElementById("vto-reconnect-btn"),
      snapshotBtn: shadowRoot.getElementById("vto-snapshot-btn"),
      toggleCamBtn: shadowRoot.getElementById("vto-toggle-cam"),
      mirrorBtn: shadowRoot.getElementById("vto-toggle-mirror"),
      minimizeBtn: shadowRoot.getElementById("vto-minimize-btn"),
      closeBtn: shadowRoot.getElementById("vto-close-btn"),
      toast: shadowRoot.getElementById("vto-toast")
    };

    // Attach Event Listeners
    makeElementDraggable(elements.header, elements.window);

    elements.minimizeBtn.addEventListener("click", toggleMinimize);
    elements.closeBtn.addEventListener("click", () => toggleModalVisibility(false));
    elements.pickClothBtn.addEventListener("click", () => toggleGarmentPicker());
    elements.topClearBtn.addEventListener("click", () => clearSlot("TOP"));
    elements.bottomClearBtn.addEventListener("click", () => clearSlot("BOTTOM"));
    elements.catAutoBtn.addEventListener("click", () => setCategoryMode("AUTO"));
    elements.catTopBtn.addEventListener("click", () => setCategoryMode("TOP"));
    elements.catBottomBtn.addEventListener("click", () => setCategoryMode("BOTTOM"));
    elements.lightingSlider.addEventListener("input", (e) => {
      setLightingIntensity(parseFloat(e.target.value) / 100.0);
    });

    elements.snapshotBtn.addEventListener("click", takeSnapshot);
    elements.reconnectBtn.addEventListener("click", () => {
      state.reconnectAttempts = 0;
      initWebSocket();
    });
    elements.toggleCamBtn.addEventListener("click", toggleWebcam);
    elements.mirrorBtn.addEventListener("click", toggleMirror);
    elements.modeAiBtn.addEventListener("click", () => setViewMode("ai"));
    if (elements.modeSkelBtn) elements.modeSkelBtn.addEventListener("click", () => setViewMode("skeleton"));
    elements.modeCamBtn.addEventListener("click", () => setViewMode("camera"));

    // Restore persisted settings from chrome.storage.local
    loadPersistedSettings();
  }

  // =========================================================================
  // 9. Toast Notification System
  // =========================================================================
  let toastTimer = null;
  function showToast(message, type = "info") {
    if (!elements.toast) return;
    if (toastTimer) clearTimeout(toastTimer);

    elements.toast.textContent = message;
    elements.toast.className = `vto-toast show ${type}`;

    toastTimer = setTimeout(() => {
      elements.toast.className = "vto-toast";
    }, 2800);
  }

  // =========================================================================
  // 10. Instant Snapshot Capture
  // =========================================================================
  function takeSnapshot() {
    if (!elements.video || !elements.canvas) return;

    try {
      const snapCanvas = document.createElement("canvas");
      snapCanvas.width = elements.canvas.width || 640;
      snapCanvas.height = elements.canvas.height || 480;
      const snapCtx = snapCanvas.getContext("2d");

      if (state.viewMode === "camera") {
        // Capture raw mirrored webcam
        snapCtx.save();
        if (state.isMirrored) {
          snapCtx.translate(snapCanvas.width, 0);
          snapCtx.scale(-1, 1);
        }
        snapCtx.drawImage(elements.video, 0, 0, snapCanvas.width, snapCanvas.height);
        snapCtx.restore();
      } else {
        // Capture AI rendered canvas output
        snapCtx.drawImage(elements.canvas, 0, 0, snapCanvas.width, snapCanvas.height);
      }

      // Add branding watermark badge
      snapCtx.fillStyle = "rgba(13, 16, 28, 0.75)";
      snapCtx.fillRect(16, snapCanvas.height - 36, 150, 24);
      snapCtx.strokeStyle = "rgba(99, 102, 241, 0.6)";
      snapCtx.strokeRect(16, snapCanvas.height - 36, 150, 24);
      snapCtx.font = "bold 11px sans-serif";
      snapCtx.fillStyle = "#ffffff";
      snapCtx.fillText("ANYWEAR VTO LIVE", 28, snapCanvas.height - 20);

      // Trigger automatic PNG download
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const link = document.createElement("a");
      link.download = `Anywear_VTO_TryOn_${timestamp}.png`;
      link.href = snapCanvas.toDataURL("image/png");
      link.click();

      showToast("📸 Try-on photo saved to downloads!", "success");
    } catch (err) {
      console.error("[Anywear VTO] Snapshot failed:", err);
      showToast("Could not capture snapshot", "error");
    }
  }

  // =========================================================================
  // 11. Settings Persistence (chrome.storage.local)
  // =========================================================================
  function persistSetting(key, val) {
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ [key]: val }).catch(() => {});
    }
  }

  function loadPersistedSettings() {
    if (!chrome.storage || !chrome.storage.local) return;

    chrome.storage.local.get([
      "vto_lightingIntensity",
      "vto_categoryMode",
      "vto_isMirrored",
      "vto_windowPos"
    ], (result) => {
      if (result.vto_lightingIntensity !== undefined) {
        state.lightingIntensity = result.vto_lightingIntensity;
        if (elements.lightingSlider) {
          elements.lightingSlider.value = Math.round(state.lightingIntensity * 100);
        }
        if (elements.sliderVal) {
          elements.sliderVal.textContent = `${Math.round(state.lightingIntensity * 100)}%`;
        }
      }

      if (result.vto_categoryMode) {
        setCategoryMode(result.vto_categoryMode);
      }

      if (result.vto_isMirrored !== undefined) {
        state.isMirrored = result.vto_isMirrored;
        if (elements.video) elements.video.classList.toggle("mirrored", state.isMirrored);
        if (elements.mirrorBtn) elements.mirrorBtn.classList.toggle("active", state.isMirrored);
      }

      if (result.vto_windowPos && elements.window) {
        const { left, top } = result.vto_windowPos;
        if (left && top) {
          elements.window.style.left = left;
          elements.window.style.top = top;
          elements.window.style.right = "auto";
        }
      }
    });
  }

  // Save drag position on drag release
  const originalMakeDraggable = makeElementDraggable;
  makeElementDraggable = function(dragHandle, targetWindow) {
    originalMakeDraggable(dragHandle, targetWindow);
    dragHandle.addEventListener("mouseup", () => {
      persistSetting("vto_windowPos", {
        left: targetWindow.style.left,
        top: targetWindow.style.top
      });
    });
  };

  // =========================================================================
  // 12. Chrome Extension Message Listener
  // =========================================================================
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "TOGGLE_MODAL") {
      toggleModalVisibility();
      sendResponse({ status: "ok", isOpen: state.isOpen });
    } else if (request.action === "TRY_ON_IMAGE") {
      if (!state.isOpen) toggleModalVisibility(true);
      if (state.isMinimized) toggleMinimize();
      if (request.srcUrl) {
        setGarment(request.srcUrl, "Context Menu Selection");
        showToast("Garment selected from right-click!", "success");
      }
      sendResponse({ status: "ok" });
    }
  });

  // =========================================================================
  // 13. Initialization Sequence
  // =========================================================================
  async function init() {
    console.log("[Anywear VTO] Initializing virtual try-on overlay...");
    const excluded = isExcludedHost();
    if (excluded) {
      state.isOpen = false;
    }

    initHostPickerOverlay();
    buildModalDom();

    if (excluded && elements.window) {
      elements.window.style.display = "none";
    }

    // Attach initial allow button click handler
    const allowBtn = shadowRoot.getElementById("vto-allow-cam-btn");
    if (allowBtn) {
      allowBtn.addEventListener("click", () => startWebcam());
    }

    // Connect WebSocket stream backend in background
    initWebSocket();

    // NOTE: startWebcam() is intentionally NOT called automatically here!
    // It is only triggered when the user actually opens the widget or clicks "Turn on Camera".
  }

  // Wait for DOM to be interactive/complete
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
