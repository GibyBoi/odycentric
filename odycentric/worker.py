"""The worker process: loads the AI model once, then cuts out photos one at a time.

The app starts this inside a guard.Job and talks to it in JSON lines. The worker
first says hello with its process id so the app can confirm it really is under
the job's limits. Only then does the app send the settings line, and nothing
heavy is loaded before it arrives. Every later line is one photo. Events go
back on stdout.
"""

import json
import os
import sys
import time


def send(**event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def describe(error):
    from PIL import UnidentifiedImageError

    from odycentric.engine import PhotoError

    text = str(error)
    if isinstance(error, MemoryError) or "alloc" in text.lower():
        return "Hit its memory limit. Raise the memory limit or close other programs."
    if isinstance(error, PhotoError):
        return text
    if isinstance(error, FileNotFoundError):
        return "File not found."
    if isinstance(error, UnidentifiedImageError):
        return "Not a photo Odycentric can read."
    return f"{type(error).__name__}: {text}"


def main():
    from odycentric import guard

    try:
        guard.join(sys.argv[1])
    except (IndexError, OSError) as error:
        send(event="failed", message=f"Could not apply the safety limits ({error}).")
        return 1
    send(event="hello", pid=os.getpid())
    line = sys.stdin.readline()
    if not line:
        return 0
    settings = json.loads(line)
    threads = int(settings["threads"])
    guard.set_own_priority(int(settings["priority"]))  # no effect if the job pins a lower one
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)

    import cv2

    from odycentric import engine

    cv2.setNumThreads(threads)
    started = time.perf_counter()
    try:
        remover = engine.Remover(settings["model"], threads)
    except Exception as error:
        send(event="failed", message=describe(error))
        return 1
    send(event="ready", seconds=round(time.perf_counter() - started, 1))

    for line in sys.stdin:
        job = json.loads(line)
        send(event="started", id=job["id"])
        started = time.perf_counter()
        try:
            result = engine.process(
                remover, job["src"], job["out_dir"], resolution=job.get("resolution", 1024),
                max_side=job.get("max_side"), sharp=job.get("sharp", False), cleanup=job.get("cleanup", True),
                background=job.get("background"))
        except Exception as error:
            send(event="error", id=job["id"], message=describe(error),
                 seconds=round(time.perf_counter() - started, 2))
            continue
        send(event="done", id=job["id"], **result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
