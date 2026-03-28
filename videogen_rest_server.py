import os

os.environ["PYTHONUTF8"] = "1"

import argparse
import base64
import enum
import logging
import queue
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

logger = logging.getLogger("videogen_rest_server")

# ---------------------------------------------------------------------------
# Default model paths (same as webui.py)
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "checkpoints/ltx-2.3-22b-distilled.safetensors"
DEFAULT_UPSAMPLER = "checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_GEMMA = "checkpoints/gemma-3-12b-it-qat-q4_0-unquantized"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobRequest(BaseModel):
    prompt: str
    height: int = Field(default=1536, ge=256, le=2048)
    width: int = Field(default=1024, ge=256, le=2048)
    duration: float = Field(default=5.0, ge=0.5, le=11.0)
    fps: int = Field(default=24, ge=1, le=60)
    seed: int = Field(default=42)
    enhance_prompt: bool = False
    start_image_base64: Optional[str] = None
    end_image_base64: Optional[str] = None


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    prompt: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus


class QueueResponse(BaseModel):
    running: Optional[str] = None
    queued: list[str] = []
    completed_pending_download: list[str] = []


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    device: str
    jobs_completed: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipeline: DistilledPipeline | None = None
_jobs: dict[str, JobInfo] = {}
_job_params: dict[str, JobRequest] = {}
_video_files: dict[str, Path] = {}
_lock = threading.Lock()
_job_queue: queue.Queue[str | None] = queue.Queue()
_temp_dir: Path | None = None
_server_start: float = 0.0
_jobs_completed: int = 0
_worker_thread: threading.Thread | None = None
_cli_args: argparse.Namespace | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seconds_to_frames(seconds: float, fps: float) -> int:
    raw = int(round(seconds * fps))
    k = max(1, round((raw - 1) / 8))
    return k * 8 + 1


def _decode_base64_image(b64_string: str) -> Path:
    data = base64.b64decode(b64_string)
    path = _temp_dir / f"{uuid.uuid4()}.png"
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _generate_video(job_id: str, req: JobRequest) -> Path:
    num_frames = _seconds_to_frames(req.duration, req.fps)

    images: list[ImageConditioningInput] = []
    temp_images: list[Path] = []

    if req.start_image_base64 is not None:
        img_path = _decode_base64_image(req.start_image_base64)
        temp_images.append(img_path)
        images.append(ImageConditioningInput(path=str(img_path), frame_idx=0, strength=1.0))

    if req.end_image_base64 is not None:
        img_path = _decode_base64_image(req.end_image_base64)
        temp_images.append(img_path)
        images.append(ImageConditioningInput(path=str(img_path), frame_idx=num_frames - 1, strength=1.0))

    tiling_config = TilingConfig.default()

    video_iter, audio = _pipeline(
        prompt=req.prompt,
        seed=req.seed,
        height=req.height,
        width=req.width,
        num_frames=num_frames,
        frame_rate=req.fps,
        images=images,
        tiling_config=tiling_config,
        enhance_prompt=req.enhance_prompt,
    )

    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    output_path = _temp_dir / f"{job_id}.mp4"

    encode_video(
        video_iter,
        fps=req.fps,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )

    # Clean up temp images
    for p in temp_images:
        p.unlink(missing_ok=True)

    return output_path


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
def _worker_loop() -> None:
    global _jobs_completed
    while True:
        job_id = _job_queue.get()
        if job_id is None:
            break

        with _lock:
            job = _jobs.get(job_id)
            if job is None or job.status == JobStatus.cancelled:
                continue
            job.status = JobStatus.running
            job.started_at = datetime.now(timezone.utc).isoformat()
            req = _job_params[job_id]

        try:
            logger.info("Starting job %s: %s", job_id, req.prompt[:80])
            output_path = _generate_video(job_id, req)
            with _lock:
                job.status = JobStatus.completed
                job.completed_at = datetime.now(timezone.utc).isoformat()
                _video_files[job_id] = output_path
                _jobs_completed += 1
            logger.info("Completed job %s -> %s", job_id, output_path)

        except Exception as e:
            with _lock:
                job.status = JobStatus.failed
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.error = str(e)
            logger.exception("Job %s failed", job_id)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _temp_dir, _server_start, _worker_thread

    args = _cli_args
    _server_start = time.monotonic()

    # Create temp directory
    _temp_dir = Path(tempfile.mkdtemp(prefix="ltx2_api_"))
    logger.info("Temp directory: %s", _temp_dir)

    # Load pipeline
    logger.info("Loading pipeline (this takes ~30 seconds)...")
    _pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.checkpoint,
        gemma_root=args.gemma,
        spatial_upsampler_path=args.upsampler,
        loras=(),
        device=torch.device("cuda"),
        quantization=QuantizationPolicy.fp8_cast(),
    )
    logger.info("Pipeline loaded!")

    # Start worker
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    _worker_thread.start()

    yield

    # Shutdown
    _job_queue.put(None)
    _worker_thread.join(timeout=5)
    if _temp_dir and _temp_dir.exists():
        shutil.rmtree(_temp_dir, ignore_errors=True)
    logger.info("Server shut down.")


app = FastAPI(title="LTX-2 Video Generation API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/jobs", status_code=202, response_model=JobSubmitResponse)
def submit_job(req: JobRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    job = JobInfo(job_id=job_id, status=JobStatus.queued, prompt=req.prompt, created_at=now)

    with _lock:
        _jobs[job_id] = job
        _job_params[job_id] = req

    _job_queue.put(job_id)
    logger.info("Queued job %s", job_id)
    return JobSubmitResponse(job_id=job_id, status=JobStatus.queued)


@app.get("/api/v1/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/jobs/{job_id}/video")
def download_video(job_id: str, background_tasks: BackgroundTasks):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != JobStatus.completed:
            raise HTTPException(status_code=409, detail=f"Job is {job.status.value}, not completed")
        video_path = _video_files.get(job_id)
        if video_path is None or not video_path.exists():
            raise HTTPException(status_code=410, detail="Video already downloaded or deleted")
        # Remove from tracking immediately to prevent double-download
        del _video_files[job_id]

    def _cleanup():
        try:
            video_path.unlink(missing_ok=True)
            logger.info("Cleaned up video file for job %s", job_id)
        except Exception:
            logger.exception("Failed to clean up video for job %s", job_id)

    background_tasks.add_task(_cleanup)
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"ltx2_{job_id[:8]}.mp4",
    )


@app.delete("/api/v1/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status == JobStatus.queued:
            job.status = JobStatus.cancelled

        video_path = _video_files.pop(job_id, None)
        if video_path and video_path.exists():
            video_path.unlink(missing_ok=True)

        del _jobs[job_id]
        _job_params.pop(job_id, None)

    return Response(status_code=204)


@app.get("/api/v1/queue", response_model=QueueResponse)
def get_queue():
    with _lock:
        running = None
        queued = []
        completed = []
        for jid, job in _jobs.items():
            if job.status == JobStatus.running:
                running = jid
            elif job.status == JobStatus.queued:
                queued.append(jid)
            elif job.status == JobStatus.completed and jid in _video_files:
                completed.append(jid)
    return QueueResponse(running=running, queued=queued, completed_pending_download=completed)


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        pipeline_loaded=_pipeline is not None,
        device=str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        jobs_completed=_jobs_completed,
        uptime_seconds=round(time.monotonic() - _server_start, 1),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="LTX-2 REST API Server")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--upsampler", default=DEFAULT_UPSAMPLER)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    _cli_args = args
    uvicorn.run(app, host=args.host, port=args.port)
