import os

os.environ["PYTHONUTF8"] = "1"

import argparse
import threading
from datetime import datetime
from pathlib import Path

import av
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
DEFAULT_OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# tqdm monkey-patching for Gradio progress
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
        prefix = getattr(_progress_state, "prefix", "")
        for item in self.iterable:
            if cb is not None and self.total:
                desc = f"{prefix}{self.label} — step {self.n + 1}/{self.total}"
                cb((self.n, self.total), desc=desc)
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
# Helper functions
# ---------------------------------------------------------------------------
pipeline: DistilledPipeline | None = None
output_dir: Path = Path(DEFAULT_OUTPUT_DIR)


def _seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert duration in seconds to the nearest valid frame count (8k+1)."""
    raw = int(round(seconds * fps))
    k = max(1, round((raw - 1) / 8))
    return k * 8 + 1


def _extract_last_frame(video_path: str) -> str | None:
    """Extract the last frame of a video and save it as a PNG."""
    if not video_path:
        return None
    container = av.open(video_path)
    stream = container.streams.video[0]
    last_frame = None
    for frame in container.decode(stream):
        last_frame = frame
    container.close()
    if last_frame is None:
        return None
    img = last_frame.to_image()
    frame_path = Path(video_path).with_suffix(".last_frame.png")
    img.save(str(frame_path))
    return str(frame_path)


def _make_output_path(suffix: str = "") -> str:
    """Generate a timestamped output path in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"ltx2_{ts}{suffix}.mp4"
    return str(output_dir / name)


# ---------------------------------------------------------------------------
# Core generation (shared by single and multi mode)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _generate_single(
    prompt: str,
    start_image: str | None,
    end_image: str | None,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    frame_rate: float,
) -> str:
    """Generate a single video. Returns the output file path."""
    images: list[ImageConditioningInput] = []
    if start_image is not None:
        images.append(ImageConditioningInput(path=start_image, frame_idx=0, strength=1.0))
    if end_image is not None:
        images.append(ImageConditioningInput(path=end_image, frame_idx=num_frames - 1, strength=1.0))

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
    out = _make_output_path()
    encode_video(
        video_iter,
        fps=frame_rate,
        audio=audio,
        output_path=out,
        video_chunks_number=video_chunks_number,
    )
    return out


# ---------------------------------------------------------------------------
# Single-video generation (tab)
# ---------------------------------------------------------------------------
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
    height, width, seed = int(height), int(width), int(seed)

    _progress_state.fn = progress
    _progress_state.tqdm_count = 0
    _progress_state.prefix = ""

    try:
        progress(0, desc=f"Generating {num_frames} frames ({duration:.1f}s)...")
        return _generate_single(prompt, start_image, end_image, height, width, num_frames, seed, frame_rate)
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("CUDA out of memory — try reducing resolution or frame count.")
    finally:
        _progress_state.fn = None
        _progress_state.tqdm_count = 0
        _progress_state.prefix = ""


# ---------------------------------------------------------------------------
# Multi-video generation (tab)
# ---------------------------------------------------------------------------
def generate_multi(
    prompts_text: str,
    start_image: str | None,
    end_image: str | None,
    height: int,
    width: int,
    duration: float,
    seed: int,
    frame_rate: float,
    progress: gr.Progress = gr.Progress(),
) -> list[str]:
    prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]
    if not prompts:
        raise gr.Error("Bitte mindestens einen Prompt eingeben (ein Prompt pro Zeile).")

    num_frames = _seconds_to_frames(duration, frame_rate)
    height, width, seed = int(height), int(width), int(seed)
    total = len(prompts)

    _progress_state.fn = progress
    _progress_state.prefix = ""

    results: list[str] = []
    current_start_image = start_image

    try:
        for i, prompt in enumerate(prompts):
            _progress_state.tqdm_count = 0
            _progress_state.prefix = f"[Video {i + 1}/{total}] "

            progress(0, desc=f"[Video {i + 1}/{total}] Starting...")

            # First video: use configured start image
            # Middle videos: use last frame of previous video
            # Last video: also use configured end image
            this_start = current_start_image
            this_end = end_image if i == total - 1 else None

            out = _generate_single(prompt, this_start, this_end, height, width, num_frames, seed + i, frame_rate)
            results.append(out)

            # Extract last frame for next video's start
            if i < total - 1:
                current_start_image = _extract_last_frame(out)

        return results

    except torch.cuda.OutOfMemoryError:
        raise gr.Error("CUDA out of memory — try reducing resolution or frame count.")
    finally:
        _progress_state.fn = None
        _progress_state.tqdm_count = 0
        _progress_state.prefix = ""


def _format_multi_results(video_paths: list[str]) -> str:
    """Format the list of generated video paths as a readable summary."""
    if not video_paths:
        return ""
    lines = [f"Generated {len(video_paths)} videos:"]
    for i, p in enumerate(video_paths, 1):
        lines.append(f"  {i}. {p}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LTX-2 Video Generator") as demo:
        gr.Markdown("# LTX-2 Video Generator")

        # Shared settings
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    height = gr.Slider(256, 2048, value=1536, step=64, label="Height")
                    width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                with gr.Row():
                    duration = gr.Slider(0.5, 11, value=5, step=0.5, label="Duration (seconds)")
                    frame_rate = gr.Slider(1, 60, value=24, step=1, label="FPS")
                seed = gr.Number(value=42, label="Seed", precision=0)

        with gr.Tabs():
            # ---- Single Video Tab ----
            with gr.Tab("Single Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your video...")
                        with gr.Row():
                            single_start = gr.Image(label="Start Image (optional)", type="filepath")
                            single_end = gr.Image(label="End Image (optional)", type="filepath")
                        single_gen_btn = gr.Button("Generate", variant="primary", size="lg")
                        single_last_frame_btn = gr.Button("Use last frame as start image", interactive=False)

                    with gr.Column(scale=1):
                        single_video = gr.Video(label="Generated Video")

                single_gen_btn.click(
                    fn=generate,
                    inputs=[single_prompt, single_start, single_end, height, width, duration, seed, frame_rate],
                    outputs=single_video,
                ).then(
                    fn=lambda: gr.update(interactive=True),
                    outputs=single_last_frame_btn,
                )

                single_last_frame_btn.click(
                    fn=_extract_last_frame,
                    inputs=single_video,
                    outputs=single_start,
                )

            # ---- Multi Video Tab ----
            with gr.Tab("Multi Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        multi_prompts = gr.Textbox(
                            label="Prompts (one per line)",
                            lines=8,
                            placeholder="Scene 1: A woman walks through a medieval market...\nScene 2: She picks up an apple and talks to the merchant...\nScene 3: She walks away into the sunset...",
                        )
                        with gr.Row():
                            multi_start = gr.Image(label="Start Image (first video)", type="filepath")
                            multi_end = gr.Image(label="End Image (last video)", type="filepath")
                        multi_gen_btn = gr.Button("Generate All", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        multi_log = gr.Textbox(label="Generated Videos", lines=6, interactive=False)
                        multi_preview = gr.Video(label="Last Generated Video")

                def _run_multi_and_preview(prompts_text, start_img, end_img, h, w, dur, s, fps, progress=gr.Progress()):
                    paths = generate_multi(prompts_text, start_img, end_img, h, w, dur, s, fps, progress)
                    log = _format_multi_results(paths)
                    last_video = paths[-1] if paths else None
                    return log, last_video

                multi_gen_btn.click(
                    fn=_run_multi_and_preview,
                    inputs=[multi_prompts, multi_start, multi_end, height, width, duration, seed, frame_rate],
                    outputs=[multi_log, multi_preview],
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2 Web UI")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--upsampler", default=DEFAULT_UPSAMPLER)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated videos")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
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
    print(f"Pipeline loaded! Videos will be saved to: {output_dir.resolve()}")

    demo = build_ui()
    demo.queue(max_size=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft())
