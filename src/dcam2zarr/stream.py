"""Core streaming logic for DCAM to Zarr."""
from typing import Optional
import time
import numpy as np
from pyDCAM import HDCAM, DCAMIDPROP
import acquire_zarr as aqz


def get_camera_config(hdcam: HDCAM) -> tuple[tuple[int, int], np.dtype]:
    """Auto-detect camera frame shape and dtype.
    
    Returns:
        ((height, width), dtype) tuple
    """
    width = int(hdcam.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_WIDTH))
    height = int(hdcam.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_HEIGHT))
    
    # Capture a test frame to get dtype
    hdcam.dcambuf_alloc(1)
    hwait = hdcam.dcamwait_open()
    hdcam.dcamcap_start()
    hwait.dcamwait_start(timeout=1000)
    test_frame = hdcam.dcambuf_copyframe()
    hdcam.dcamcap_stop()
    hdcam.dcambuf_release()
    
    return (height, width), test_frame.dtype


class DCAMStreamer:
    """Stream frames from a DCAM camera to Zarr format.
    
    Args:
        hdcam: Initialized HDCAM camera instance
        output_path: Path where Zarr store will be written
        chunk_time: Number of frames per time chunk
        max_frames: Maximum frames to capture (None = unlimited)
    """
    
    def __init__(
        self,
        hdcam: HDCAM,
        output_path: str,
        chunk_time: int = 64,
        max_frames: Optional[int] = None,
    ):
        self.hdcam = hdcam
        self.output_path = output_path
        self.chunk_time = chunk_time
        self.max_frames = max_frames
        
        # Auto-detect camera configuration
        self.frame_shape, self.dtype = get_camera_config(hdcam)
        
        self._setup_zarr_stream()
        self.frames_written = 0
        self.bytes_written = 0
        self.start_time = None
        
    def _setup_zarr_stream(self):
        """Initialize Zarr stream with frame and timestamp arrays."""
        height, width = self.frame_shape
        
        settings = aqz.StreamSettings(
            store_path=self.output_path,
            overwrite=True
        )
        
        # Main frame array
        frame_array = aqz.ArraySettings(
            output_key="frames",
            data_type=self.dtype,
            dimensions=[
                aqz.Dimension(
                    name="t",
                    kind=aqz.DimensionType.TIME,
                    array_size_px=0,  # Unlimited
                    chunk_size_px=self.chunk_time,
                    shard_size_chunks=1
                ),
                aqz.Dimension(
                    name="y",
                    kind=aqz.DimensionType.SPACE,
                    array_size_px=height,
                    chunk_size_px=height,
                    shard_size_chunks=1
                ),
                aqz.Dimension(
                    name="x",
                    kind=aqz.DimensionType.SPACE,
                    array_size_px=width,
                    chunk_size_px=width,
                    shard_size_chunks=1
                )
            ]
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
                    chunk_size_px=self.chunk_time,
                    shard_size_chunks=1
                )
            ]
        )
        
        settings.arrays = [frame_array] #, timestamp_array]
        self.stream = aqz.ZarrStream(settings)
        
    def capture_frame(self, hwait) -> tuple[np.ndarray, float]:
        """Capture a single frame with timestamp.
        
        Returns:
            (frame_data, timestamp) tuple
        """
        hwait.dcamwait_start(timeout=1000)
        timestamp = time.time()
        frame = self.hdcam.dcambuf_copyframe()
        
        return frame, timestamp
        
    def stream_frames(self):
        """Main streaming loop."""
        self.hdcam.dcambuf_alloc(1)
        hwait = self.hdcam.dcamwait_open()
        
        self.start_time = time.time()
        
        try:
            self.hdcam.dcamcap_start()
            
            while True:
                if self.max_frames and self.frames_written >= self.max_frames:
                    break
                    
                frame, timestamp = self.capture_frame(hwait)
                
                # Write frame (needs [1, H, W] shape for single frame)
                self.stream.append(frame[np.newaxis, :, :], key="frames")
                
                # Write timestamp
                # self.stream.append(np.array([timestamp]), key="timestamps")
                
                self.frames_written += 1
                self.bytes_written += frame.nbytes
                
        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        finally:
            self.hdcam.dcamcap_stop()
            self.hdcam.dcambuf_release()
            self.stream.close()
    
    def get_stats(self) -> dict:
        """Return capture statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            "frames_captured": self.frames_written,
            "bytes_written": self.bytes_written,
            "elapsed_seconds": elapsed,
            "frames_per_second": self.frames_written / elapsed if elapsed > 0 else 0,
            "throughput_mbps": (self.bytes_written / elapsed / 1e6) if elapsed > 0 else 0,
        }