import asyncio
import time
import cv2
import numpy as np
import websockets

async def benchmark_websocket_latency(num_frames=20):
    uri = "ws://localhost:8000/ws/stream"
    print(f"Connecting to {uri}...")
    
    # Create synthetic test frame matching downscaled resolution (480x360)
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.putText(img, "TEST FRAME", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
    _, encoded = cv2.imencode(".jpg", img, encode_params)
    frame_bytes = encoded.tobytes()
    print(f"Frame payload size: {len(frame_bytes)} bytes ({len(frame_bytes)/1024:.1f} KB)")

    latencies = []

    async with websockets.connect(uri) as ws:
        # Warm-up (MediaPipe model load on first frame)
        print("Sending warm-up frame...")
        t0 = time.perf_counter()
        await ws.send(frame_bytes)
        resp = await ws.recv()
        warmup_time = (time.perf_counter() - t0) * 1000
        print(f"Warm-up round-trip: {warmup_time:.1f} ms")

        # In-Flight Lock simulation (ping-pong benchmark)
        print(f"\nBenchmarking {num_frames} frames with ping-pong in-flight lock...")
        for i in range(num_frames):
            start = time.perf_counter()
            await ws.send(frame_bytes)
            resp = await ws.recv()
            rtt = (time.perf_counter() - start) * 1000
            latencies.append(rtt)
            print(f"Frame #{i+1:02d}: RTT = {rtt:.2f} ms (response size: {len(resp)} bytes)")
            await asyncio.sleep(0.02) # simulate ~30-40 FPS camera feed cadence

    avg_rtt = sum(latencies) / len(latencies)
    min_rtt = min(latencies)
    max_rtt = max(latencies)
    p95_rtt = sorted(latencies)[int(len(latencies) * 0.95)]

    print("\n" + "="*50)
    print("LATENCY BENCHMARK RESULTS")
    print("="*50)
    print(f"Average RTT : {avg_rtt:.2f} ms")
    print(f"Minimum RTT : {min_rtt:.2f} ms")
    print(f"Maximum RTT : {max_rtt:.2f} ms")
    print(f"95th %ile   : {p95_rtt:.2f} ms")
    print(f"Target (<50ms): {'PASSED [OK]' if avg_rtt < 50 else 'FAILED'}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(benchmark_websocket_latency(25))
