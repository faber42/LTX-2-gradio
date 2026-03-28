"""Client for the LTX-2 Video Generation REST API.

No external dependencies — uses only the Python standard library.
"""

import argparse
import base64
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request


def _get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode()}


def _download(url: str, output_path: str, timeout: float = 300.0) -> int:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(output_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    return os.path.getsize(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a video via the LTX-2 REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --prompt "A cat walking through a garden"
  %(prog)s --prompt "A sunset over the ocean" --duration 3.0 --output sunset.mp4
  %(prog)s --prompt "A person waving" --start-image photo.png --host 192.168.1.100
  %(prog)s --prompt "Slow motion water" --height 1024 --width 1536 --fps 24 --seed 123
""",
    )
    parser.add_argument("--prompt", required=True, help="Text description of the video")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output file path (default: output.mp4)")
    parser.add_argument("--host", default="localhost", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=8020, help="Server port (default: 8020)")
    parser.add_argument("--height", type=int, default=None, help="Video height in pixels (default: 1536)")
    parser.add_argument("--width", type=int, default=None, help="Video width in pixels (default: 1024)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: 5.0)")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second (default: 24)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument("--enhance-prompt", action="store_true", help="Let Gemma rewrite the prompt")
    parser.add_argument("--start-image", default=None, help="Path to start image (first frame)")
    parser.add_argument("--end-image", default=None, help="Path to end image (last frame)")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval in seconds (default: 5)")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}/api/v1"

    # Health check
    try:
        info = _get_json(f"{base}/health", timeout=5)
        print(f"Server: {info['device']} | Jobs completed: {info['jobs_completed']}")
    except (urllib.error.URLError, OSError):
        print(f"ERROR: Cannot connect to {args.host}:{args.port}. Is the server running?")
        sys.exit(1)

    # Build request
    payload: dict = {"prompt": args.prompt}
    if args.height is not None:
        payload["height"] = args.height
    if args.width is not None:
        payload["width"] = args.width
    if args.duration is not None:
        payload["duration"] = args.duration
    if args.fps is not None:
        payload["fps"] = args.fps
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.enhance_prompt:
        payload["enhance_prompt"] = True

    if args.start_image:
        if not os.path.isfile(args.start_image):
            print(f"ERROR: Start image not found: {args.start_image}")
            sys.exit(1)
        with open(args.start_image, "rb") as f:
            payload["start_image_base64"] = base64.b64encode(f.read()).decode()

    if args.end_image:
        if not os.path.isfile(args.end_image):
            print(f"ERROR: End image not found: {args.end_image}")
            sys.exit(1)
        with open(args.end_image, "rb") as f:
            payload["end_image_base64"] = base64.b64encode(f.read()).decode()

    # Submit job
    status_code, data = _post_json(f"{base}/jobs", payload)
    if status_code != 202:
        print(f"ERROR: Server returned {status_code}: {data}")
        sys.exit(1)
    job_id = data["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll for completion
    last_status = None
    while True:
        data = _get_json(f"{base}/jobs/{job_id}")
        status = data["status"]

        if status != last_status:
            print(f"  Status: {status}")
            last_status = status

        if status == "completed":
            break
        if status == "failed":
            print(f"  Error: {data.get('error')}")
            sys.exit(1)

        time.sleep(args.poll_interval)

    # Download video
    size = _download(f"{base}/jobs/{job_id}/video", args.output)
    print(f"Saved: {args.output} ({size:,} bytes)")


if __name__ == "__main__":
    main()
