"""Core streaming logic for DCAM to Zarr."""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Optional, Tuple

import numpy as np

try:
    import pyDCAM
except AttributeError:  # module 'ctypes' has no attribute 'windll'
    logging.warning("Failed to import pyDCAM")
import acquire_zarr as aqz

from .config import Compression, Multiscale
from .shm import FrameBuffer


def get_shm_name(camera_index: int) -> str:
    """Generate a unique shared memory name for a given camera index."""
    return f"dcam2zarr_cam{camera_index}"


class Streamer(ABC):
    """Abstract class for camera streaming."""

    def __init__(
        self,
        camera_index: int,
        output_path: str,
        chunk_x: int,
        chunk_y: int,
        chunk_t: int,
        shard_x: int,
        shard_y: int,
        shard_t: int,
        max_frames: Optional[int] = None,
        compression: Optional[Compression] = None,
        multiscale: Optional[Multiscale] = None,
        enable_http: bool = False,
    ):
        self.camera_index = camera_index
        self.output_path = output_path
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.chunk_t = chunk_t
        self.shard_x = shard_x
        self.shard_y = shard_y
        self.shard_t = shard_t
        self.max_frames = max_frames
        self.compression = compression
        self.multiscale = multiscale
        self.enable_http = enable_http
        self.frame_buffer = None

        self.frames_written = 0
        self.bytes_written = 0
        self.start_time = None

        # set by subclasses
        self.frame_shape: Tuple[int, int] | None = None
        self.dtype: np.dtype | None = None

        # Initialize camera and get frame properties
        self._initialize_camera()

    @abstractmethod
    def _initialize_camera(self) -> None:
        """Initialize camera and detect frame shape/dtype."""
        ...

    @abstractmethod
    def _start_capture(self) -> None:
        """Start camera capture."""
        ...

    @abstractmethod
    def _stop_capture(self) -> None:
        """Stop camera capture."""
        ...

    @abstractmethod
    def _capture_frame(self) -> Tuple[np.ndarray, float]:
        """Capture a single frame with timestamp.

        Returns:
            (frame_data, timestamp) tuple
        """
        ...

    def _setup_zarr_stream(self):
        """Initialize Zarr stream with frame and timestamp arrays."""
        height, width = self.frame_shape

        settings = aqz.StreamSettings(store_path=self.output_path, overwrite=True)

        compression = None
        if self.compression and self.compression.enabled:
            compression = aqz.CompressionSettings(
                compressor=aqz.Compressor.BLOSC1,
                codec=(
                    aqz.CompressionCodec.BLOSC_ZSTD
                    if self.compression.codec.lower() == "zstd"
                    else aqz.CompressionCodec.BLOSC_LZ4
                ),
                level=self.compression.level,
                shuffle=1,
            )

        downsampling_method = None
        if self.multiscale and self.multiscale.enabled:
            match self.multiscale.method.lower():
                case "mean":
                    downsampling_method = aqz.DownsamplingMethod.MEAN
                case "min":
                    downsampling_method = aqz.DownsamplingMethod.MIN
                case "max":
                    downsampling_method = aqz.DownsamplingMethod.MAX
                case _:
                    raise ValueError(
                        f"Unsupported downsampling method: {self.multiscale.method}"
                    )

        # Main frame array
        frame_array = aqz.ArraySettings(
            output_key="frames",
            data_type=self.dtype,
            compression=compression,
            downsampling_method=downsampling_method,
            dimensions=[
                aqz.Dimension(
                    name="t",
                    kind=aqz.DimensionType.TIME,
                    array_size_px=0,  # Unlimited
                    chunk_size_px=self.chunk_t,
                    shard_size_chunks=self.shard_t,
                ),
                aqz.Dimension(
                    name="y",
                    kind=aqz.DimensionType.SPACE,
                    array_size_px=height,
                    chunk_size_px=self.chunk_y or height,
                    shard_size_chunks=self.shard_y,
                ),
                aqz.Dimension(
                    name="x",
                    kind=aqz.DimensionType.SPACE,
                    array_size_px=width,
                    chunk_size_px=self.chunk_x or width,
                    shard_size_chunks=self.shard_x,
                ),
            ],
        )

        # Timestamp array (system time for now)
        timestamp_array = aqz.ArraySettings(
            output_key="timestamps",
            data_type=np.float64,
            dimensions=[
                aqz.Dimension(
                    name="t",
                    kind=aqz.DimensionType.TIME,
                    array_size_px=0,
                    chunk_size_px=self.chunk_t,
                    shard_size_chunks=self.shard_t,
                )
            ],
        )

        settings.arrays = [frame_array]  # , timestamp_array]
        self.stream = aqz.ZarrStream(settings)

    def _setup_shared_memory(self):
        """Setup shared memory buffer for HTTP server if enabled."""
        if self.enable_http:
            shm_name = get_shm_name(self.camera_index)
            downsampled_shape = tuple(s // 4 for s in self.frame_shape)

            self.frame_buffer = FrameBuffer(
                name=shm_name,
                shape=downsampled_shape,
                dtype=self.dtype,
                create=True,
            )

    def stream_frames(self):
        """Main streaming loop."""
        # Setup Zarr and shared memory
        print("Setting up Zarr stream...")
        self._setup_zarr_stream()
        self._setup_shared_memory()
        print("Setup complete. Starting capture...")

        self.start_time = time.time()

        try:
            self._start_capture()

            while True:
                if self.max_frames and self.frames_written >= self.max_frames:
                    break

                frame, timestamp = self._capture_frame()

                # Write to Zarr
                self.stream.append(frame, key="frames")

                # Write to shared memory for HTTP server
                if self.frame_buffer is not None and self.frames_written % 5 == 0: # downsample every 5 frames to reduce load
                    small = frame[::4, ::4]
                    self.frame_buffer.write(small, self.frames_written, timestamp)

                self.frames_written += 1
                self.bytes_written += frame.nbytes

        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        finally:
            self._stop_capture()
            self.stream.close()

            if self.frame_buffer is not None:
                self.frame_buffer.close()
                self.frame_buffer.unlink()

    def get_stats(self) -> dict:
        """Return capture statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            "frames_captured": self.frames_written,
            "bytes_written": self.bytes_written,
            "elapsed_seconds": elapsed,
            "frames_per_second": self.frames_written / elapsed if elapsed > 0 else 0,
            "throughput_mbps": (
                (self.bytes_written / elapsed / 1e6) if elapsed > 0 else 0
            ),
        }


class DCAMStreamer(Streamer):
    """Stream frames from a DCAM camera to Zarr format.

    Args:
        hdcam: Initialized HDCAM camera instance
        output_path: Path where Zarr store will be written
        chunk_time: Number of frames per time chunk
        max_frames: Maximum frames to capture (None = unlimited)
    """

    def __init__(
        self,
        hdcam: Any,
        camera_index: int,
        output_path: str,
        chunk_x: int,
        chunk_y: int,
        chunk_t: int,
        shard_x: int,
        shard_y: int,
        shard_t: int,
        max_frames: Optional[int] = None,
        compression: Optional[Compression] = None,
        multiscale: Optional[Multiscale] = None,
        enable_http: bool = False,
    ):
        self.hdcam = hdcam
        self.hwait = None

        super().__init__(
            camera_index,
            output_path,
            chunk_x,
            chunk_y,
            chunk_t,
            shard_x,
            shard_y,
            shard_t,
            max_frames,
            compression,
            multiscale,
            enable_http,
        )

    def _initialize_camera(self):
        width = int(
            self.hdcam.dcamprop_getvalue(pyDCAM.DCAMIDPROP.DCAM_IDPROP_IMAGE_WIDTH)
        )
        height = int(
            self.hdcam.dcamprop_getvalue(pyDCAM.DCAMIDPROP.DCAM_IDPROP_IMAGE_HEIGHT)
        )

        # Capture a test frame to get dtype
        self.hdcam.dcambuf_alloc(1)
        hwait = self.hdcam.dcamwait_open()
        self.hdcam.dcamcap_start()
        hwait.dcamwait_start(timeout=1000)
        test_frame = self.hdcam.dcambuf_copyframe()
        self.hdcam.dcamcap_stop()
        self.hdcam.dcambuf_release()

        self.frame_shape = (height, width)
        self.dtype = test_frame.dtype

    def _start_capture(self):
        self.hdcam.dcambuf_alloc(1)
        self.hwait = self.hdcam.dcamwait_open()
        self.hdcam.dcamcap_start()

    def _stop_capture(self):
        self.hdcam.dcamcap_stop()
        self.hdcam.dcambuf_release()
        self.stream.close()

    def _capture_frame(self):
        self.hwait.dcamwait_start(timeout=1000)
        timestamp = time.time()
        frame = self.hdcam.dcambuf_copyframe()

        return frame, timestamp


class DummyStreamer(Streamer):
    def __init__(
        self,
        camera_index: int,
        output_path: str,
        chunk_x: int,
        chunk_y: int,
        chunk_t: int,
        shard_x: int,
        shard_y: int,
        shard_t: int,
        max_frames: Optional[int] = None,
        compression: Optional[Compression] = None,
        multiscale: Optional[Multiscale] = None,
        enable_http: bool = False,
    ):
        super().__init__(
            camera_index,
            output_path,
            chunk_x,
            chunk_y,
            chunk_t,
            shard_x,
            shard_y,
            shard_t,
            max_frames,
            compression,
            multiscale,
            enable_http,
        )

        # will be initialized in _start_capture
        self.center_y = None
        self.center_x = None
        self.y_grid = None
        self.x_grid = None
        self.radius_grid = None

    def _initialize_camera(self):
        self.frame_shape = (2304, 4096)
        self.dtype = np.dtype("uint16")

    def _start_capture(self):
        """Initialize radial pattern parameters."""
        height, width = self.frame_shape

        # Center of the pattern
        self.center_y = height / 2
        self.center_x = width / 2

        # Create coordinate grids
        y = np.arange(height)
        x = np.arange(width)
        self.x_grid, self.y_grid = np.meshgrid(x, y)

        # Pre-compute radius from center
        self.radius_grid = np.sqrt(
            (self.x_grid - self.center_x) ** 2 + (self.y_grid - self.center_y) ** 2
        )

    def _stop_capture(self):
        """No-op"""
        pass

    def _capture_frame(self):
        """Generate a radial sine pattern frame.

        Pattern: intensity = sin(r * freq + phase)
        where phase is driven by timestamp for animation

        Returns:
            (frame_data, timestamp) tuple
        """
        timestamp = time.time()

        # Use timestamp to create animated pattern
        phase = timestamp * 2.0  # 2 rad/s rotation
        frequency = 0.05  # Spatial frequency

        # Radial sine pattern
        pattern = np.sin(self.radius_grid * frequency + phase)

        # Scale to dtype range
        if self.dtype == np.uint16:
            # Map [-1, 1] -> [0, 65535]
            frame = ((pattern + 1.0) / 2.0 * 65535).astype(np.uint16)
        elif self.dtype == np.uint8:
            # Map [-1, 1] -> [0, 255]
            frame = ((pattern + 1.0) / 2.0 * 255).astype(np.uint8)
        else:
            # For float types, keep in [-1, 1]
            frame = pattern.astype(self.dtype)

        return frame, timestamp
