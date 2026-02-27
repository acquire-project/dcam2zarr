# dcam2zarr

Prototype for streaming Hamamatsu DCAM cameras to Zarr format using acquire-zarr, with optional HTTP server for live preview.

## Features

- Single-camera streaming to Zarr (OME-Zarr compatible)
- Configurable chunking and sharding
- Optional compression (Blosc LZ4/ZSTD)
- Optional multiscale pyramids
- HTTP server for live frame preview with ROI support
- Performance monitoring and statistics

## Installation
```bash
uv sync
```

## Usage

### Basic Streaming

Stream a single camera using command-line arguments:
```bash
# Stream camera 0, unlimited frames
uv run dcam2zarr 0 --output data.zarr

# Capture 1000 frames with custom chunking
uv run dcam2zarr 0 --output test.zarr --frames 1000 --chunk-t 128 --chunk-x 32
```

### Configuration File

For complex setups, use a JSON configuration file:
```json
{
  "camera_index": 0,
  "output_path": "data.zarr",
  "max_frames": null,
  
  "chunking": {
    "x": null,
    "y": null,
    "t": 64
  },
  
  "sharding": {
    "x": 1,
    "y": 1,
    "t": 1
  },
  
  "compression": {
    "enabled": true,
    "codec": "zstd",
    "level": 5
  },
  
  "multiscale": {
    "enabled": false,
    "method": "mean"
  },
  
  "http_server": {
    "enabled": true,
    "port": 8080,
    "host": "0.0.0.0"
  }
}
```

Run with config file:
```bash
uv run dcam2zarr --config config.json
```

CLI arguments override config file values:
```bash
uv run dcam2zarr --config config.json --frames 500 --output test.zarr
```

## HTTP Live Preview

When `http_server.enabled` is true, a separate HTTP server process provides live frame access.

### Web Viewer

Open in browser:
```
http://localhost:8080/viewer
```

Features:
- Live frame updates
- Configurable update rate
- Auto-scaling or manual intensity range
- ROI selection and preview
- Real-time statistics

### API Endpoints

**GET /latest** - Get full latest frame (raw binary)
```bash
curl "http://localhost:8080/latest" --output frame.bin
```

**GET /latest?x=X&y=Y&w=W&h=H** - Get ROI from latest frame
```bash
curl "http://localhost:8080/latest?x=100&y=100&w=512&h=512" --output roi.bin
```

**GET /info** - Get frame metadata (JSON)
```bash
curl "http://localhost:8080/info"
```

Response headers:
- `X-Frame-Number`: Sequential frame number
- `X-Timestamp`: Unix timestamp
- `X-Frame-Shape`: Height,Width
- `X-Frame-Dtype`: NumPy dtype string

### Python Client

Test the HTTP server programmatically:
```python
import requests
import numpy as np

# Fetch full frame
resp = requests.get("http://localhost:8080/latest")
shape = tuple(map(int, resp.headers["X-Frame-Shape"].split(",")))
dtype = np.dtype(resp.headers["X-Frame-Dtype"])
frame = np.frombuffer(resp.content, dtype=dtype).reshape(shape)

# Fetch ROI
resp = requests.get("http://localhost:8080/latest", 
                    params={"x": 100, "y": 100, "w": 256, "h": 256})
shape = tuple(map(int, resp.headers["X-Frame-Shape"].split(",")))
roi = np.frombuffer(resp.content, dtype=dtype).reshape(shape)
```

## Utility Commands

List available cameras:
```bash
uv run ls_cams
```

## Performance

Typical performance on Quest 2 cameras (2304×4096 uint16):

- **Without HTTP server**: ~109 fps, 2.06 GB/s
- **With HTTP server** (4× downsampled preview): ~104 fps, 1.96 GB/s
- **With compression** (Blosc ZSTD level 5): Variable, typically 2-4× reduction

Performance depends on:
- Disk I/O capability (NVMe recommended for multi-camera setups)
- Chunk/shard configuration
- Compression settings
- System memory bandwidth

## Multi-Camera Setup

For simultaneous multi-camera acquisition, run separate processes:
```bash
# Terminal 1
uv run dcam2zarr 0 --output cam0.zarr --config cam0_config.json

# Terminal 2
uv run dcam2zarr 1 --output cam1.zarr --config cam1_config.json
```

Each camera can have its own HTTP server on different ports.

## Output Format

Creates Zarr stores with:
- **frames** array: shape (t, y, x), configurable dtype
- Optional multiscale levels (if enabled)
- OME-Zarr metadata

Zarr format is V3, compatible with napari, OME tools, and zarr-python.

## Development
```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black src
```

## Architecture
```
[Camera Process]                    [HTTP Server Process]
    |                                       |
DCAM API                                    |
    |                                       |
    +---> acquire-zarr (Zarr write)         |
    |                                       |
    +---> shared memory -----------------> HTTP API
          (downsampled frames)              |
                                            v
                                       Web clients
```

The HTTP server runs in a separate process and reads from shared memory to avoid blocking acquisition.

## Known Limitations

- HTTP preview is downsampled 4× to reduce memory bandwidth
- Shared memory buffer holds only the latest frame
- No frame buffering for missed HTTP requests
- Windows only (pyDCAM limitation)
- Live viewer expects frames to be served on localhost:8080