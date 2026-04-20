"""File-backed mmap client for IPC with the C++ engine.

Uses a regular file (~/.slmfs/ipc_shm.bin) mapped with mmap instead of
POSIX shared memory (shm_open). This bypasses macOS launchd sandbox
isolation — file-backed mmap is visible to all processes regardless of
Mach bootstrap namespace.

Safety model:
- Single Python producer assumed (FUSE runs with nothreads=True, or slmfs add
  runs standalone). Multiple concurrent Python producers are NOT supported.
- The C++ engine is the sole consumer of the SPSC queue.
- Slab acquire: only Python acquires (clears bits). C++ only releases (sets bits
  via atomic fetch_or). This means the Python load-check-store on the bitmask is
  safe because there is no concurrent acquirer.
- Slab release: Python sets bits (for READ response acknowledgement). C++ also
  sets bits. Both are idempotent OR operations on disjoint bits, so no conflict.
"""

import fcntl
import mmap
import os
import struct
import time
from pathlib import Path
from typing import Optional

from .config import SlmfsConfig
from .cooker import DONE_MAGIC


class ShmClient:
    """Python-side file-backed mmap access: slab allocation + SPSC push.

    IMPORTANT: Only one Python process should use this at a time.
    Running FUSE and slmfs-add simultaneously against the same shared
    memory is not supported (would require a real interprocess mutex).
    """

    # Control block layout (must match C++ ControlBlock):
    # [0..8)     atomic<uint64_t> free_bitmask
    # [64..72)   head (atomic, cache-line padded)
    # [128..136) cached_tail
    # [192..200) tail (atomic)
    # [256..264) cached_head
    # [320..1344) buffer[256] (uint32_t each)
    _BITMASK_OFF = 0
    _RING_HEAD_OFF = 64
    _RING_TAIL_OFF = 192
    _RING_BUF_OFF = 320
    _RING_CAPACITY = 256
    _RING_MASK = _RING_CAPACITY - 1

    def __init__(self, config: SlmfsConfig):
        shm_path = Path(config.shm_path)

        # Enforce single-producer: acquire an exclusive file lock.
        lock_path = shm_path.with_suffix(".lock")
        self._lock_file = open(lock_path, "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_file.write(str(os.getpid()))
            self._lock_file.flush()
        except OSError:
            self._lock_file.close()
            raise RuntimeError(
                f"Another SLMFS producer is already connected to '{shm_path}'. "
                f"Only one Python producer (FUSE or slmfs-add) may run at a time."
            )

        # Open the file-backed shared memory region.
        # The C++ engine creates and truncates this file; we open read-write.
        if not shm_path.exists():
            raise FileNotFoundError(
                f"Shared memory file not found: {shm_path}\n"
                f"Is the SLMFS engine running? "
                f"Start it with: slmfs_engine --db-path=~/.slmfs/memory.db"
            )

        self._fd = os.open(str(shm_path), os.O_RDWR)
        self._mm = mmap.mmap(self._fd, config.shm_size)
        self._slab_size = config.slab_size
        self._slab_count = config.slab_count
        self._ctrl_size = config.control_block_size
        self._data_offset = config.control_block_size

    def close(self):
        """Release the mmap, file descriptor, and producer lock."""
        self._mm.close()
        os.close(self._fd)
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()

    def _read_u64(self, offset: int) -> int:
        return struct.unpack_from("<Q", self._mm, offset)[0]

    def _write_u64(self, offset: int, value: int):
        struct.pack_into("<Q", self._mm, offset, value)

    def acquire_slab(self) -> Optional[int]:
        """Acquire a free slab. Returns slab index or None if pool exhausted."""
        current = self._read_u64(self._BITMASK_OFF)
        if current == 0:
            return None

        idx = (current & -current).bit_length() - 1
        new_val = current & ~(1 << idx)
        self._write_u64(self._BITMASK_OFF, new_val)
        return idx

    def release_slab(self, index: int):
        """Return a slab to the free pool."""
        current = self._read_u64(self._BITMASK_OFF)
        new_val = current | (1 << index)
        self._write_u64(self._BITMASK_OFF, new_val)

    def write_to_slab(self, index: int, payload: bytes):
        """Copy payload into slab."""
        offset = self._data_offset + index * self._slab_size
        self._mm[offset : offset + len(payload)] = payload

    def read_slab(self, index: int, length: int) -> bytes:
        """Read bytes from a slab."""
        offset = self._data_offset + index * self._slab_size
        return bytes(self._mm[offset : offset + length])

    def read_slab_u32(self, index: int, byte_offset: int = 0) -> int:
        """Read a uint32 from a slab at the given byte offset."""
        offset = self._data_offset + index * self._slab_size + byte_offset
        return struct.unpack_from("<I", self._mm, offset)[0]

    def push_handle(self, handle: int) -> bool:
        """Push 32-bit handle to SPSC ring buffer.

        Returns False if the ring buffer is full (caller should retry).
        """
        head = self._read_u64(self._RING_HEAD_OFF)
        next_head = (head + 1) & self._RING_MASK

        tail = self._read_u64(self._RING_TAIL_OFF)
        if next_head == tail:
            return False

        slot_off = self._RING_BUF_OFF + (head & self._RING_MASK) * 4
        struct.pack_into("<I", self._mm, slot_off, handle)

        self._write_u64(self._RING_HEAD_OFF, next_head)
        return True

    def push_handle_blocking(
        self, handle: int, timeout: float = 5.0, retry_delay: float = 0.001
    ) -> bool:
        """Push handle with retry on full ring. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.push_handle(handle):
                return True
            time.sleep(retry_delay)
        return False

    def wait_for_done(
        self, slab_index: int, timeout: float = 1.0
    ) -> Optional[bytes]:
        """Spin-wait for engine to write DONE magic, then read and release slab."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            magic = self.read_slab_u32(slab_index, 0)
            if magic == DONE_MAGIC:
                text_length = self.read_slab_u32(slab_index, 20)
                result = self.read_slab(slab_index, 64 + text_length)
                self.release_slab(slab_index)
                return result[64 : 64 + text_length]
            time.sleep(0.0001)
        self.release_slab(slab_index)
        return None
