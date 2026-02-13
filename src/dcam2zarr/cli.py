"""Command-line interface for DCAM to Zarr streaming."""
import argparse
from pathlib import Path
import pyDCAM
from .stream import DCAMStreamer


def main():
    parser = argparse.ArgumentParser(
        description="Stream Hamamatsu DCAM camera to Zarr format"
    )
    parser.add_argument(
        "camera_index",
        type=int,
        help="Camera device index (0, 1, ...)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.zarr"),
        help="Output directory for Zarr store (default: output.zarr)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to capture (default: unlimited, stop with Ctrl-C)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of frames per time chunk (default: 64)"
    )
    
    args = parser.parse_args()
    
    # Initialize DCAM API
    device_count = pyDCAM.dcamapi_init()
    
    if args.camera_index >= device_count:
        print(f"Error: Camera {args.camera_index} not found (found {device_count} devices)")
        pyDCAM.dcamapi_uninit()
        return 1
    
    try:
        with pyDCAM.HDCAM(args.camera_index) as hdcam:
            # Get camera info
            model = hdcam.dcamdev_getstring(pyDCAM.DCAM_IDSTR.DCAM_IDSTR_MODEL)
            camera_id = hdcam.dcamdev_getstring(pyDCAM.DCAM_IDSTR.DCAM_IDSTR_CAMERAID)
            
            print(f"Streaming from: {model} ({camera_id})")
            print(f"Output: {args.output}")
            print(f"Frames: {'unlimited' if args.frames is None else args.frames}")
            print(f"Chunk size: {args.chunk_size}")
            print()
            
            streamer = DCAMStreamer(
                hdcam=hdcam,
                output_path=str(args.output),
                chunk_time=args.chunk_size,
                max_frames=args.frames
            )
            
            print(f"Frame shape: {streamer.frame_shape}")
            print(f"Data type: {streamer.dtype}")
            print("Starting capture... (Ctrl-C to stop)")
            print()
            
            streamer.stream_frames()
            
            # Print statistics
            stats = streamer.get_stats()
            print("\nCapture complete:")
            print(f"  Frames: {stats['frames_captured']}")
            print(f"  Bytes: {stats['bytes_written']:,}")
            print(f"  Time: {stats['elapsed_seconds']:.2f}s")
            print(f"  FPS: {stats['frames_per_second']:.2f}")
            print(f"  Throughput: {stats['throughput_mbps']:.2f} MB/s")
            
    finally:
        pyDCAM.dcamapi_uninit()
    
    return 0


if __name__ == "__main__":
    exit(main())