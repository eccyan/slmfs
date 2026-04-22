"""Online bulk ingestion into a running engine.

Usage: python -m slmfs add <file.md> [file2.md ...] [--shm-path=path]
       python -m slmfs add --stop-fuse ~/docs/**/*.md

Bypasses FUSE — streams chunks directly via shared memory with
backpressure handling when the slab pool is temporarily full.

The --stop-fuse flag automatically stops the FUSE service before
ingestion and restarts it afterward (requires launchctl on macOS
or systemctl on Linux).
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .config import SlmfsConfig
from .cooker import cook_write, CMD_WRITE_COMMIT
from .embedder import MiniLMEmbedder
from .init import parse_markdown, Chunk
from .shm_client import ShmClient


def _stop_fuse() -> bool:
    """Stop the FUSE service. Returns True if it was running."""
    if platform.system() == "Darwin":
        plist = Path.home() / "Library/LaunchAgents/com.eccyan.slmfs-fuse.plist"
        if not plist.exists():
            return False
        result = subprocess.run(
            ["launchctl", "unload", str(plist)],
            capture_output=True,
        )
        return result.returncode == 0
    else:
        result = subprocess.run(
            ["systemctl", "--user", "stop", "slmfs-fuse"],
            capture_output=True,
        )
        return result.returncode == 0


def _start_fuse():
    """Restart the FUSE service."""
    if platform.system() == "Darwin":
        plist = Path.home() / "Library/LaunchAgents/com.eccyan.slmfs-fuse.plist"
        if plist.exists():
            subprocess.run(["launchctl", "load", str(plist)], capture_output=True)
    else:
        subprocess.run(
            ["systemctl", "--user", "start", "slmfs-fuse"],
            capture_output=True,
        )


def add_file(
    path: Path,
    embedder: MiniLMEmbedder,
    shm: ShmClient,
    max_retries: int = 100,
    retry_delay: float = 0.05,
):
    """Parse, embed, and stream a Markdown file into the running engine."""
    chunks = parse_markdown(path)
    if not chunks:
        print(f"  No chunks found in {path}")
        return 0

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts)

    ingested = 0
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = cook_write(
            chunk.text, embedding, parent_id=0, depth=chunk.depth
        )

        slab_idx = None
        for attempt in range(max_retries):
            slab_idx = shm.acquire_slab()
            if slab_idx is not None:
                break
            time.sleep(retry_delay)

        if slab_idx is None:
            print(
                f"  WARNING: slab pool exhausted after {max_retries} retries, "
                f"skipping chunk {i}"
            )
            continue

        shm.write_to_slab(slab_idx, payload)
        handle = (CMD_WRITE_COMMIT << 24) | slab_idx
        if not shm.push_handle_blocking(handle):
            print(f"  WARNING: ring buffer full, releasing slab for chunk {i}")
            shm.release_slab(slab_idx)
            continue
        ingested += 1

    return ingested


def main():
    """Entry point: python -m slmfs add <file.md> [--stop-fuse] [--shm-path=path]"""
    config = SlmfsConfig()
    paths: list[Path] = []
    stop_fuse = False

    for arg in sys.argv[1:]:
        if arg.startswith("--shm-path="):
            config.shm_path = Path(arg.split("=", 1)[1]).expanduser()
        elif arg == "--stop-fuse":
            stop_fuse = True
        else:
            paths.append(Path(arg))

    if not paths:
        print("Usage: python -m slmfs add <file.md> [file2.md ...]")
        print("       python -m slmfs add --stop-fuse ~/docs/**/*.md")
        sys.exit(1)

    fuse_was_running = False
    if stop_fuse:
        print("Stopping FUSE service...")
        fuse_was_running = _stop_fuse()
        if fuse_was_running:
            time.sleep(1)  # wait for lock release

    try:
        print(f"Connecting to engine via shm: {config.shm_path}")
        embedder = MiniLMEmbedder()
        shm = ShmClient(config)

        total = 0
        for p in paths:
            print(f"  Ingesting {p}...")
            count = add_file(p, embedder, shm)
            print(f"    {count} chunks streamed")
            total += count

        shm.close()
        print(f"Done. {total} chunks ingested into running engine.")
    finally:
        if fuse_was_running:
            print("Restarting FUSE service...")
            _start_fuse()


if __name__ == "__main__":
    main()
