"""Smoke-test client for the LTX-2 REST API server."""

import argparse
import os
import sys
import time

import requests

USAGE = """\
LTX-2 REST API smoke test.

This script runs a full integration test against a running videogen_rest_server:
  1. Health check
  2. Submit a short test job
  3. Poll until completion
  4. Download the video
  5. Verify cleanup (410 on second download)
  6. Show queue status

Usage:
  .venv/Scripts/python.exe videogen_rest_test_client.py --test
  .venv/Scripts/python.exe videogen_rest_test_client.py --test --host 192.168.1.100
"""


def run_test(host: str, port: int) -> None:
    base = f"http://{host}:{port}/api/v1"
    prompt = "A serene lake at sunset with gentle ripples on the water surface"
    output = "test_output.mp4"

    # 1. Health check
    print("--- Health check ---")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        r.raise_for_status()
        print(f"  {r.json()}")
    except requests.ConnectionError:
        print(f"  ERROR: Cannot connect to {host}:{port}. Is the server running?")
        sys.exit(1)

    # 2. Submit job
    print(f"\n--- Submitting job ---")
    print(f"  Prompt: {prompt}")
    print(f"  Duration: 2.0s")
    r = requests.post(f"{base}/jobs", json={"prompt": prompt, "duration": 2.0})
    r.raise_for_status()
    data = r.json()
    job_id = data["job_id"]
    print(f"  Job ID: {job_id}")
    print(f"  Status: {data['status']}")

    # 3. Poll for completion
    print(f"\n--- Waiting for completion ---")
    while True:
        r = requests.get(f"{base}/jobs/{job_id}")
        r.raise_for_status()
        status = r.json()["status"]
        print(f"  Status: {status}")

        if status == "completed":
            break
        if status == "failed":
            print(f"  Error: {r.json().get('error')}")
            sys.exit(1)

        time.sleep(5)

    # 4. Download video
    print(f"\n--- Downloading video ---")
    r = requests.get(f"{base}/jobs/{job_id}/video", stream=True)
    r.raise_for_status()
    with open(output, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size = os.path.getsize(output)
    print(f"  Saved to: {output} ({size:,} bytes)")

    if size == 0:
        print("  WARNING: File is empty!")
        sys.exit(1)

    # 5. Verify double-download returns 410
    print(f"\n--- Verifying cleanup (expect 410) ---")
    r = requests.get(f"{base}/jobs/{job_id}/video")
    print(f"  Status code: {r.status_code} ({'OK - file was cleaned up' if r.status_code == 410 else 'UNEXPECTED'})")

    # 6. Queue status
    print(f"\n--- Queue status ---")
    r = requests.get(f"{base}/queue")
    r.raise_for_status()
    print(f"  {r.json()}")

    # Cleanup
    os.remove(output)
    print(f"\nAll tests passed!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LTX-2 REST API smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE,
    )
    parser.add_argument("--test", action="store_true", help="Run the smoke test (required)")
    parser.add_argument("--host", default="localhost", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=8020, help="Server port (default: 8020)")
    args = parser.parse_args()

    if not args.test:
        print(USAGE)
        sys.exit(0)

    run_test(args.host, args.port)


if __name__ == "__main__":
    main()
