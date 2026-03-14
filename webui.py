import os

os.environ["PYTHONUTF8"] = "1"

import argparse
import tempfile
import threading
from pathlib import Path

import gradio as gr
import torch

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

# ---------------------------------------------------------------------------
# Default model paths (relative to repo root)
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "checkpoints/ltx-2.3-22b-distilled.safetensors"
DEFAULT_UPSAMPLER = "checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_GEMMA = "checkpoints/gemma-3-12b-it-qat-q4_0-unquantized"

# ---------------------------------------------------------------------------
# B. tqdm monkey-patching for Gradio progress
# ---------------------------------------------------------------------------
import ltx_pipelines.utils.media_io as _media_io_module
import ltx_pipelines.utils.samplers as _samplers_module

_progress_state = threading.local()

_STAGE_LABELS = [
    "Stage 1: Denoising",
    "Stage 2: Upscaling",
    "Decoding video",
]


class _GradioTqdm:
    """Drop-in tqdm replacement that forwards progress to Gradio."""

    def __init__(self, iterable=None, total=None, **_kwargs: object) -> None:
        self.iterable = iterable
        self.total = total or (len(iterable) if hasattr(iterable, "__len__") else None)
        self.n = 0

        idx = getattr(_progress_state, "tqdm_count", 0)
        _progress_state.tqdm_count = idx + 1
        self.label = _STAGE_LABELS[idx] if idx < len(_STAGE_LABELS) else f"Processing ({idx})"

    def __iter__(self):
        cb = getattr(_progress_state, "fn", None)
        for item in self.iterable:
            if cb is not None and self.total:
                cb((self.n, self.total), desc=f"{self.label} — step {self.n + 1}/{self.total}")
            yield item
            self.n += 1

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        pass


def _patch_tqdm() -> None:
    _samplers_module.tqdm = _GradioTqdm
    _media_io_module.tqdm = _GradioTqdm


# ---------------------------------------------------------------------------
# C. Generation function
# ---------------------------------------------------------------------------
pipeline: DistilledPipeline | None = None


def _seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert duration in seconds to the nearest valid frame count (8k+1)."""
    raw = int(round(seconds * fps))
    # Snap to nearest 8k+1 (minimum 9)
    k = max(1, round((raw - 1) / 8))
    return k * 8 + 1


@torch.inference_mode()
def generate(
    prompt: str,
    start_image: str | None,
    end_image: str | None,
    height: int,
    width: int,
    duration: float,
    seed: int,
    frame_rate: float,
    progress: gr.Progress = gr.Progress(),
) -> str:
    if not prompt or not prompt.strip():
        raise gr.Error("Bitte einen Prompt eingeben.")

    num_frames = _seconds_to_frames(duration, frame_rate)
    height = int(height)
    width = int(width)
    seed = int(seed)

    # Set up progress callback
    _progress_state.fn = progress
    _progress_state.tqdm_count = 0

    try:
        # Image conditioning (start and/or end frame)
        images: list[ImageConditioningInput] = []
        if start_image is not None:
            images.append(ImageConditioningInput(path=start_image, frame_idx=0, strength=1.0))
        if end_image is not None:
            images.append(ImageConditioningInput(path=end_image, frame_idx=num_frames - 1, strength=1.0))

        progress(0, desc=f"Generating {num_frames} frames ({duration:.1f}s)...")

        tiling_config = TilingConfig.default()
        video_iter, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=False,
        )

        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        output_path = tempfile.mktemp(suffix=".mp4", prefix="ltx2_")
        encode_video(
            video_iter,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

        return output_path

    except torch.cuda.OutOfMemoryError:
        raise gr.Error("CUDA out of memory — try reducing resolution or frame count.")
    finally:
        _progress_state.fn = None
        _progress_state.tqdm_count = 0


# ---------------------------------------------------------------------------
# D. Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LTX-2 Video Generator") as demo:
        gr.Markdown("# LTX-2 Video Generator")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your video...")

                with gr.Row():
                    start_image = gr.Image(label="Start Image (optional)", type="filepath")
                    end_image = gr.Image(label="End Image (optional)", type="filepath")

                with gr.Row():
                    height = gr.Slider(256, 2048, value=1536, step=64, label="Height")
                    width = gr.Slider(256, 2048, value=1024, step=64, label="Width")

                with gr.Row():
                    duration = gr.Slider(0.5, 11, value=5, step=0.5, label="Duration (seconds)")
                    frame_rate = gr.Slider(1, 60, value=24, step=1, label="FPS")

                seed = gr.Number(value=42, label="Seed", precision=0)
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video")

        generate_btn.click(
            fn=generate,
            inputs=[prompt, start_image, end_image, height, width, duration, seed, frame_rate],
            outputs=video_output,
        )

    return demo


# ---------------------------------------------------------------------------
# E. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2 Web UI")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--upsampler", default=DEFAULT_UPSAMPLER)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    _patch_tqdm()

    print("Loading pipeline (this takes ~30 seconds)...")
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.checkpoint,
        gemma_root=args.gemma,
        spatial_upsampler_path=args.upsampler,
        loras=(),
        device=torch.device("cuda"),
        quantization=QuantizationPolicy.fp8_cast(),
    )
    print("Pipeline loaded!")

    demo = build_ui()
    demo.queue(max_size=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft())
