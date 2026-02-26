"""Command-line interface for DCAM to Zarr streaming."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

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
        help="Path to JSON configuration file (if provided, other args override config values)",
    )

    # required if no config file
    parser.add_argument(
        "camera_index", type=int, nargs="?", help="Camera device index (0, 1, ...)"
    )

    # optional overrides
    parser.add_argument("--output", type=Path, help="Output directory for Zarr stores")
    parser.add_argument("--frames", type=int, help="Number of frames to capture")
    parser.add_argument("--chunk-x", type=int, help="X chunk size in pixels")
    parser.add_argument("--chunk-y", type=int, help="Y chunk size in pixels")
    parser.add_argument("--chunk-t", type=int, help="T chunk size in pixels")

    args = parser.parse_args()
    if args.config:
        config = DCAMConfig.from_json_file(args.config)
    else:
        if args.camera_index is None:
            parser.error("camera_index is required when --config is not provided")
        config = DCAMConfig(
            camera_index=args.camera_index,
            output_path=args.output or "output.zarr",
            max_frames=args.frames,
        )

    # CLI overrides
    if args.camera_index is not None:
        config.camera_index = args.camera_index
    if args.output is not None:
        config.output_path = args.output
    if args.frames is not None:
        config.max_frames = args.frames
    if args.chunk_x is not None:
        config.chunking.x = args.chunk_x
    if args.chunk_y is not None:
        config.chunking.y = args.chunk_y
    if args.chunk_t is not None:
        config.chunking.t = args.chunk_t

    # Initialize DCAM API
    device_count = pyDCAM.dcamapi_init()

    if config.camera_index >= device_count:
        print(
            f"Error: Camera {config.camera_index} not found (found {device_count} devices)"
        )
        pyDCAM.dcamapi_uninit()
        return 1

    try:
        with pyDCAM.HDCAM(config.camera_index) as hdcam:
            # Get camera info
            model = hdcam.dcamdev_getstring(pyDCAM.DCAM_IDSTR.DCAM_IDSTR_MODEL)
            camera_id = hdcam.dcamdev_getstring(pyDCAM.DCAM_IDSTR.DCAM_IDSTR_CAMERAID)

            print(f"Streaming from: {model} ({camera_id})")
            print(f"Output: {config.output_path}")
            print(
                f"Frames: {'unlimited' if config.max_frames is None else config.max_frames}"
            )
            print(
                f"Chunking: x={config.chunking.x}, y={config.chunking.y}, t={config.chunking.t}"
            )
            print(
                f"Sharding: x={config.sharding.x}, y={config.sharding.y}, t={config.sharding.t}"
            )
            print(
                f"Compression: {config.compression.codec.value} (level {config.compression.level})"
                if config.compression.enabled
                else "Compression: disabled"
            )
            print(
                f"Multiscale: {config.multiscale.method.value}"
                if config.multiscale.enabled
                else "Multiscale: disabled"
            )
            print()

            streamer = DCAMStreamer(
                hdcam=hdcam,
                camera_index=config.camera_index,
                output_path=str(config.output_path),
                chunk_x=config.chunking.x,
                chunk_y=config.chunking.y,
                chunk_t=config.chunking.t,
                shard_x=config.sharding.x,
                shard_y=config.sharding.y,
                shard_t=config.sharding.t,
                max_frames=config.max_frames,
                compression=config.compression if config.compression.enabled else None,
                multiscale=config.multiscale if config.multiscale.enabled else None,
                enable_http=config.http_server.enabled,
            )

            print(f"Frame shape: {streamer.frame_shape}")
            print(f"Data type: {streamer.dtype}")
            print("Starting capture... (Ctrl-C to stop)")
            print()

            server_process = None
            if config.http_server.enabled:
                print(
                    f"Launching server with height {streamer.frame_shape[0] // 4} and width {streamer.frame_shape[1] // 4}"
                )
                server_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "dcam2zarr.server",
                        "--camera-index",
                        str(config.camera_index),
                        "--height",
                        str(
                            streamer.frame_shape[0] // 4
                        ),  # downsampled for HTTP preview
                        "--width",
                        str(
                            streamer.frame_shape[1] // 4
                        ),  # downsampled for HTTP preview
                        "--dtype",
                        str(streamer.dtype),
                        "--port",
                        str(config.http_server.port),
                        "--host",
                        config.http_server.host,
                    ]
                )
                print(
                    f"HTTP server started on {config.http_server.host}:{config.http_server.port}"
                )

            try:
                streamer.stream_frames()
            finally:
                if server_process:
                    server_process.terminate()
                    server_process.wait()

            # Get compressed size on disk
            table_size = (
                config.sharding.x * config.sharding.y * config.sharding.t * 8 * 2 + 4
            )  # Approximate overhead per shard
            bytes_on_disk = 0

            data_dir = (
                config.output_path / "frames/0"
                if config.multiscale.enabled
                else config.output_path / "frames"
            )
            for dirpath, _, filenames in os.walk(data_dir):
                for f in filenames:
                    if f.endswith(".zarr"):
                        continue

                    fp = os.path.join(dirpath, f)
                    bytes_on_disk += (
                        os.path.getsize(fp) - table_size
                    )  # Subtract overhead

            # Print statistics
            stats = streamer.get_stats()
            print("\nCapture complete:")
            print(f"  Frames: {stats['frames_captured']}")
            print(f"  Bytes: {stats['bytes_written']:,}")
            print(
                f"  Bytes on Disk: {bytes_on_disk:,}"
                if config.compression.enabled
                else f"  Bytes on Disk: {bytes_on_disk:,}"
            )
            print(
                f"  Compression Ratio: {stats['bytes_written'] / bytes_on_disk:.2f}"
                if bytes_on_disk > 0
                else "  Compression Ratio: N/A"
            )
            print(f"  Time: {stats['elapsed_seconds']:.2f}s")
            print(f"  FPS: {stats['frames_per_second']:.2f}")
            print(f"  Throughput: {stats['throughput_mbps']:.2f} MB/s")

    finally:
        pyDCAM.dcamapi_uninit()

    return 0


def ls_cams():
    device_count = pyDCAM.dcamapi_init()
    print(f"Found {device_count} camera(s):")
    for i in range(device_count):
        with pyDCAM.HDCAM(i) as hdcam:
            model = hdcam.dcamdev_getstring(pyDCAM.DCAM_IDSTR.DCAM_IDSTR_MODEL)
            camera_id = hdcam.dcamdev_getstring(pyDCAM.DCAM_IDSTR.DCAM_IDSTR_CAMERAID)
            print(f"  [{i}] {model} ({camera_id})")

    pyDCAM.dcamapi_uninit()


if __name__ == "__main__":
    exit(main())
