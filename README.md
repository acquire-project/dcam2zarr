# dcam2zarr

Minimal prototype for streaming Hamamatsu DCAM cameras to Zarr format using acquire-zarr.

## Installation
```bash
uv sync
```

## Usage

Stream a single camera:
```bash
uv run dcam2zarr <camera_index> --output <path.zarr> --frames <n>
```

Examples:
```bash
# Stream camera 0, unlimited frames (Ctrl-C to stop)
uv run dcam2zarr 0 --output data.zarr

# Capture 1000 frames from camera 1
uv run dcam2zarr 1 --output test.zarr --frames 1000

# Test different chunk sizes
uv run dcam2zarr 0 --output test.zarr --frames 500 --chunk-size 32
```

## Output

Creates a Zarr store with a single array `frames` containing captured images with shape `(t, y, x)`.

## Performance Testing

This tool is designed to test streaming performance. Key metrics reported:
- Frames per second
- Throughput (MB/s)
- Total frames and bytes written

Use these to determine hardware requirements for multi-camera simultaneous streaming.