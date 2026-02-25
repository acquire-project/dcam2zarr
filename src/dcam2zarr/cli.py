"""Command-line interface for DCAM to Zarr streaming."""
import argparse
from pathlib import Path
import pyDCAM

from .stream import DCAMStreamer
from .config import DCAMConfig


def main():
    parser = argparse.ArgumentParser(
        description="Stream Hamamatsu DCAM camera to Zarr format"
    )

    # config file or individual args
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file (if provided, other args override config values)"
    )

    # required if no config file
    parser.add_argument(
        "camera_index",
        type=int,
        nargs="?",
        help="Camera device index (0, 1, ...)"
    )

    # optional overrides
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for Zarr stores"
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="Number of frames to capture"
    )
    parser.add_argument(
        "--chunk-x",
        type=int,
        help="X chunk size in pixels"
    )
    parser.add_argument(
        "--chunk-y",
        type=int,
        help="Y chunk size in pixels"
    )
    parser.add_argument(
        "--chunk-t",
        type=int,
        help="T chunk size in pixels"
    )

    args = parser.parse_args()
    if args.config:
        config = DCAMConfig.from_json_file(args.config)
    else:
        if args.camera_index is None:
            parser.error("camera_index is required when --config is not provided")
        config = DCAMConfig(camera_index=args.camera_index,
                            output_path=args.output or "output.zarr",
                            max_frames=args.frames)

    # CLI overrides
    if args.camera_index is not None:
        config.camera_index = args.camera_index
    if args.output is not None:
        config.output_path = args.output
    if args.frames is not None:
        config.frames = args.frames
    if args.chunk_x is not None:
        config.chunking.x = args.chunk_x
    if args.chunk_y is not None:
        config.chunking.y = args.chunk_y
    if args.chunk_t is not None:
        config.chunking.t = args.chunk_t

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
            print(f"Output: {config.output_path}")
            print(f"Frames: {'unlimited' if config.max_frames is None else config.max_frames}")
            print(f"Chunking: x={config.chunking.x}, y={config.chunking.y}, t={config.chunking.t}")
            print(f"Sharding: x={config.sharding.x}, y={config.sharding.y}, t={config.sharding.t}")
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
