"""Shared memory interface for frame sharing between processes."""

from multiprocessing import shared_memory
from typing import Tuple
import numpy as np
import struct


class FrameBuffer:
    """Shared memory buffer for a single frame with metadata.

    Layout:
    [8 bytes: frame_number (uint64)]
    [8 bytes: timestamp (float64)]
    [4 bytes: height (uint32)]
    [4 bytes: width (uint32)]
    [2 bytes: dtype code (uint16)]
    [remaining: frame data]
    """

    HEADER_SIZE = 26  # bytes

    def __init__(
            self, name: str, shape: Tuple[int, int], dtype: np.dtype, create: bool = False
    ):
        """
        Args:
            name: Shared memory segment name
            shape: (height, width) of frames
            dtype: Frame data type
            create: If True, create new segment; if False, attach to existing
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype

        frame_size = shape[0] * shape[1] * dtype.itemsize
        total_size = self.HEADER_SIZE + frame_size

        if create:
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=total_size
            )
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)

    def write(self, frame: np.ndarray, frame_number: int, timestamp: float) -> None:
        """Write frame with metadata to shared memory."""
        if frame.shape != self.shape:
            raise ValueError(
                f"Frame shape {frame.shape} doesn't match buffer shape {self.shape}"
            )
        if frame.dtype != self.dtype:
            raise ValueError(
                f"Frame dtype {frame.dtype} doesn't match buffer dtype {self.dtype}"
            )

        # Pack header
        header = struct.pack(
            "<QdIIH",  # Little-endian: uint64, float64, uint32, uint32, uint16
            frame_number,
            timestamp,
            self.shape[0],
            self.shape[1],
            self._dtype_to_code(self.dtype),
        )

        height, width = frame.shape
        frame_size_bytes = height * width * np.dtype(self.dtype).itemsize

        # Write header + frame data
        self.shm.buf[: self.HEADER_SIZE] = header
        self.shm.buf[self.HEADER_SIZE:self.HEADER_SIZE + frame_size_bytes] = frame.tobytes()

    def read(self) -> Tuple[np.ndarray, int, float]:
        """Read frame with metadata from shared memory.

        Returns:
            (frame, frame_number, timestamp)
        """
        # Unpack header
        header = struct.unpack("<QdIIH", bytes(self.shm.buf[: self.HEADER_SIZE]))
        frame_number, timestamp, height, width, dtype_code = header

        # Reconstruct dtype and shape
        dtype = self._code_to_dtype(dtype_code)
        shape = (height, width)

        expected_size = height * width * np.dtype(dtype).itemsize

        # Read frame data
        frame_bytes = bytes(
            self.shm.buf[self.HEADER_SIZE: self.HEADER_SIZE + expected_size]
        )
        frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape).copy()

        return frame, frame_number, timestamp

    def close(self) -> None:
        """Close shared memory (but don't unlink)."""
        self.shm.close()

    def unlink(self) -> None:
        """Unlink shared memory (destroys it)."""
        self.shm.unlink()

    @staticmethod
    def _dtype_to_code(dtype: np.dtype) -> int:
        """Convert numpy dtype to uint16 code."""
        dtype_map = {
            np.uint8: 0,
            np.uint16: 1,
            np.uint32: 2,
            np.int8: 3,
            np.int16: 4,
            np.int32: 5,
            np.float32: 6,
            np.float64: 7,
        }
        return dtype_map.get(dtype.type, 0)

    @staticmethod
    def _code_to_dtype(code: int) -> np.dtype:
        """Convert uint16 code to numpy dtype."""
        code_map = {
            0: np.uint8,
            1: np.uint16,
            2: np.uint32,
            3: np.int8,
            4: np.int16,
            5: np.int32,
            6: np.float32,
            7: np.float64,
        }
        return np.dtype(code_map.get(code, np.uint8))
