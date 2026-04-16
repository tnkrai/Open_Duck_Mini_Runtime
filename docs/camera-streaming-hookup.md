# Camera Streaming — Integration Plan

## Current State

The robot has a working Pi camera (`mini_bdx_runtime/camera.py`) that captures 512x512 JPEG frames. It's only used by `fc_test.py` (a GPT-4o vision agent that takes snapshots to navigate). The camera is **not connected** to the telemetry pipeline, the TNKR server, or the dashboard.

### What exists

| Component | File | Camera Status |
|-----------|------|--------------|
| Camera module | `mini_bdx_runtime/camera.py` | Works — captures, rotates, encodes to base64 |
| Duck config | `mini_bdx_runtime/duck_config.py` | Has `expression_features.camera` flag (default: `False`) |
| Walk script | `scripts/v2_rl_walk_mujoco.py` | **Does not read** `duck_config.camera` — flag is ignored |
| TNKR server | `scripts/tnkr_server.py` | **No camera endpoint** |
| Telemetry broadcast | `mini_bdx_runtime/cloud_publisher.py` | **No camera frames** in the payload |
| Dashboard viewer | `tnkr-dashboard` | **No camera feed UI** |

## What Needs to Be Hooked Up

### 1. Walk Script — Initialize camera when enabled

**File:** `scripts/v2_rl_walk_mujoco.py`

The walk script already reads `duck_config.camera` but never acts on it. Add camera initialization alongside the other expression features (eyes, projector, etc.):

```python
# After the antennas init block (~line 168)
if self.duck_config.camera:
    try:
        from mini_bdx_runtime.camera import Cam
        self.cam = Cam()
    except Exception as e:
        print(f"[Expression] Camera init failed, disabling: {e}")
        self.duck_config.camera = False
```

### 2. Walk Script — Include camera frames in telemetry

**File:** `scripts/v2_rl_walk_mujoco.py` (inside `run()` loop, ~line 384)

Add camera frame to the `state_snapshot` dict. Capture at a lower rate than joint data (e.g., every 500ms instead of every 20ms) to avoid bandwidth issues:

```python
# Add a frame counter or timer for camera throttling
if self.duck_config.camera and (i % 25 == 0):  # ~2 FPS at 50Hz control
    try:
        state_snapshot["camera_frame"] = self.cam.get_encoded_image()
    except Exception:
        pass  # Don't let camera failures interrupt walking
```

**Consideration:** Base64 JPEG frames at 512x512 are ~50-100KB each. At 2 FPS over Supabase Realtime, this is ~100-200KB/s. May need to:
- Reduce resolution (256x256)
- Use lower JPEG quality
- Send camera on a separate channel to avoid slowing joint telemetry

### 3. TNKR Server — Add camera snapshot endpoint

**File:** `scripts/tnkr_server.py`

Add an HTTP endpoint for on-demand snapshots (useful even without streaming):

```python
@app.get("/api/camera/snapshot")
def camera_snapshot():
    """Capture and return a single camera frame."""
    try:
        from mini_bdx_runtime.camera import Cam
        cam = Cam()
        image_b64 = cam.get_encoded_image()
        return {"image": image_b64}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Camera error: {e}")
```

### 4. Dashboard — Telemetry hook

**File:** `tnkr-dashboard/app/hooks/useSupabaseTelemetry.ts`

Extract `camera_frame` from the telemetry payload and expose it:

```typescript
// In the telemetry frame handler
if (state.camera_frame) {
    setCameraFrame(`data:image/jpeg;base64,${state.camera_frame}`);
}
```

### 5. Dashboard — Camera feed UI

**File:** `tnkr-dashboard/app/(dashboard-pages)/explore/build/[projectId]/viewer/page.tsx`

Add a picture-in-picture or sidebar panel showing the camera feed next to the URDF viewer.

### 6. Camera module — Performance improvement

**File:** `mini_bdx_runtime/camera.py`

The current `get_encoded_image()` writes to disk then reads back for base64. Encode directly from memory:

```python
def get_encoded_image(self) -> str:
    im = self.cam.capture_array()
    im = cv2.resize(im, (512, 512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    _, buffer = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')
```

## Bandwidth Considerations

| Resolution | JPEG Quality | Approx Size | At 2 FPS |
|-----------|-------------|-------------|----------|
| 512x512 | 90 | ~80KB | 160 KB/s |
| 512x512 | 70 | ~50KB | 100 KB/s |
| 256x256 | 70 | ~15KB | 30 KB/s |

**Recommendation:** Use a separate Supabase channel for camera frames (e.g., `robot-camera-{session}`) so that camera latency/bandwidth doesn't affect joint telemetry at 10Hz. The dashboard can subscribe to both channels independently.

## Duck Config

Enable camera in `~/duck_config.json`:

```json
{
  "expression_features": {
    "camera": true,
    "eyes": true,
    "projector": false,
    "speaker": false,
    "antennas": false
  }
}
```
