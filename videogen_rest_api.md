# LTX-2 Video Generation REST API

REST API for generating videos with the LTX-2 model. Jobs are queued and processed sequentially on the GPU. Generated videos are temporary — they are deleted after download.

## Starting the Server

```bash
uv run python videogen_rest_server.py
```

The server listens on `0.0.0.0:8020` by default. Optional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8020` | Port |
| `--checkpoint` | `checkpoints/ltx-2.3-22b-distilled.safetensors` | Model checkpoint |
| `--upsampler` | `checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | Spatial upsampler |
| `--gemma` | `checkpoints/gemma-3-12b-it-qat-q4_0-unquantized` | Gemma text encoder |

FastAPI auto-docs are available at `http://<host>:8020/docs`.

---

## Endpoints

### POST /api/v1/jobs

Submit a video generation job.

**Request body (JSON):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | yes | — | Text description of the video |
| `height` | int | no | 1536 | Video height in pixels (256–2048, divisible by 64) |
| `width` | int | no | 1024 | Video width in pixels (256–2048, divisible by 64) |
| `duration` | float | no | 5.0 | Duration in seconds (0.5–11.0) |
| `fps` | int | no | 24 | Frames per second (1–60) |
| `seed` | int | no | 42 | Random seed for reproducibility |
| `enhance_prompt` | bool | no | false | Let Gemma rewrite the prompt for better results |
| `start_image_base64` | string | no | null | Base64-encoded PNG/JPEG for the first frame |
| `end_image_base64` | string | no | null | Base64-encoded PNG/JPEG for the last frame |

**Response `202 Accepted`:**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued"
}
```

**Example (curl):**

```bash
curl -X POST http://localhost:8020/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking through a garden", "duration": 3.0}'
```

**Example (Python):**

```python
import requests

r = requests.post("http://localhost:8020/api/v1/jobs", json={
    "prompt": "A cat walking through a garden",
    "duration": 3.0,
})
job_id = r.json()["job_id"]
```

---

### GET /api/v1/jobs/{job_id}

Check the status of a job.

**Response `200`:**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "prompt": "A cat walking through a garden",
  "created_at": "2026-03-28T14:30:00+00:00",
  "started_at": null,
  "completed_at": null,
  "error": null
}
```

**Status values:** `queued`, `running`, `completed`, `failed`, `cancelled`

**Errors:** `404` if job not found.

---

### GET /api/v1/jobs/{job_id}/video

Download the generated video. The file is deleted from the server after a successful download.

**Response `200`:** Binary `video/mp4` stream.

**Errors:**
- `404` — Job not found
- `409` — Job not yet completed (still queued/running/failed)
- `410` — Video already downloaded or deleted

**Example (curl):**

```bash
curl -o output.mp4 http://localhost:8020/api/v1/jobs/a1b2c3d4-.../video
```

**Example (Python):**

```python
r = requests.get(f"http://localhost:8020/api/v1/jobs/{job_id}/video", stream=True)
with open("output.mp4", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
```

---

### DELETE /api/v1/jobs/{job_id}

Cancel a queued job or delete a completed job (and its video file).

**Response:** `204 No Content`

**Errors:** `404` if job not found.

---

### GET /api/v1/queue

View the current queue state.

**Response `200`:**

```json
{
  "running": "a1b2c3d4-...",
  "queued": ["e5f6g7h8-..."],
  "completed_pending_download": ["i9j0k1l2-..."]
}
```

---

### GET /api/v1/health

Server health and status.

**Response `200`:**

```json
{
  "status": "ok",
  "pipeline_loaded": true,
  "device": "NVIDIA GeForce RTX 4090",
  "jobs_completed": 5,
  "uptime_seconds": 3600.0
}
```

---

## Typical Client Flow

1. `POST /api/v1/jobs` — submit a job, receive `job_id`
2. Poll `GET /api/v1/jobs/{job_id}` every few seconds until `status` is `completed` or `failed`
3. `GET /api/v1/jobs/{job_id}/video` — download the video (file is deleted on the server after this)

A ready-to-use test client is provided: `videogen_rest_test_client.py`.

```bash
uv run python videogen_rest_test_client.py
uv run python videogen_rest_test_client.py --host 192.168.1.100 --prompt "A dog running"
```

## Notes

- Only one video is generated at a time (GPU constraint). Additional requests are queued in FIFO order.
- Videos are stored temporarily and deleted after download. If a client fails to download, the files remain until the server is restarted.
- No authentication is required. The server is intended for use within a trusted local network.
- Height and width should be divisible by 64 for the two-stage distilled pipeline.
