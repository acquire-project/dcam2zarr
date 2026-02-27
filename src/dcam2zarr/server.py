"""HTTP server for accessing latest camera frames."""

import argparse
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from .shm import FrameBuffer
from .stream import get_shm_name


def create_app(
    camera_index: int, frame_shape: tuple[int, int], dtype: np.dtype
) -> FastAPI:
    """Create FastAPI app for a specific camera.

    Args:
        camera_index: Camera device index
        frame_shape: (height, width) of frames
        dtype: Frame data type
    """
    app = FastAPI(title=f"DCAM Camera {camera_index} Server")

    # Connect to shared memory
    shm_name = get_shm_name(camera_index)
    max_retries = 50
    retry_delay = 0.1  # seconds
    
    frame_buffer = None
    for attempt in range(max_retries):
        try:
            frame_buffer = FrameBuffer(name=shm_name, shape=frame_shape, dtype=dtype, create=False)
            print(f"Connected to shared memory: {shm_name}")
            break
        except FileNotFoundError:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to connect to shared memory '{shm_name}' after {max_retries} attempts")
            time.sleep(retry_delay)

    @app.get("/")
    async def root():
        """API information."""
        return {
            "camera_index": camera_index,
            "frame_shape": frame_shape,
            "dtype": str(dtype),
            "endpoints": {
                "/latest": "Get latest full frame (raw binary)",
                "/latest/roi": "Get ROI from latest frame (query params: x, y, w, h)",
                "/info": "Get latest frame metadata",
            },
        }

    @app.get("/latest")
    async def get_latest_frame(
        x: Optional[int] = None,
        y: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None,
    ):
        """Get latest frame or ROI as raw binary.

        Query parameters (all required for ROI, or none for full frame):
            x: ROI starting x coordinate
            y: ROI starting y coordinate
            w: ROI width
            h: ROI height
        """
        try:
            frame, frame_number, timestamp = frame_buffer.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read frame: {e}")

        shape = frame.shape

        # Check if ROI parameters provided
        roi_params = [x, y, w, h]
        if any(p is not None for p in roi_params):
            # All ROI params must be provided
            if not all(p is not None for p in roi_params):
                raise HTTPException(
                    status_code=400,
                    detail="All ROI parameters (x, y, w, h) must be provided together",
                )

            # Validate ROI bounds
            height, width = frame.shape
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise HTTPException(status_code=400, detail="Invalid ROI parameters")
            if x + w > width or y + h > height:
                raise HTTPException(status_code=400, detail="ROI exceeds frame bounds")

            shape = (h, w)

            # Extract ROI
            roi = frame[y : y + h, x : x + w]
            data = roi.tobytes()
        else:
            # Full frame
            data = frame.tobytes()

        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "X-Frame-Number": str(frame_number),
                "X-Timestamp": str(timestamp),
                "X-Frame-Shape": f"{shape[0]},{shape[1]}",
                "X-Frame-Dtype": str(frame.dtype),
            },
        )

    @app.get("/info")
    async def get_frame_info():
        """Get metadata about the latest frame without returning frame data."""
        try:
            frame, frame_number, timestamp = frame_buffer.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read frame: {e}")

        return JSONResponse(
            {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
            }
        )

    @app.get("/viewer")
    async def get_viewer():
        """Serve the HTML viewer."""
        return FileResponse(
            os.path.join(os.path.dirname(__file__), "static", "viewer.html")
        )

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on server shutdown."""
        frame_buffer.close()
        # Don't unlink - camera process owns it
        # Just unregister from resource tracker
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(frame_buffer.shm._name, "shared_memory")
        except Exception:
            pass  # Best effort

    return app


def run_server(
    camera_index: int,
    frame_shape: tuple[int, int],
    dtype: np.dtype,
    port: int,
    host: str = "0.0.0.0",
):
    """Run the HTTP server.

    Args:
        camera_index: Camera device index
        frame_shape: (height, width) of frames
        dtype: Frame data type
        port: Port to bind to
        host: Host address to bind to
    """
    import uvicorn

    app = create_app(camera_index, frame_shape, dtype)
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="DCAM frame HTTP server")
    parser.add_argument("--camera-index", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    # Convert dtype string to numpy dtype
    dtype = np.dtype(args.dtype)
    frame_shape = (args.height, args.width)

    print(f"Starting HTTP server for camera {args.camera_index}")
    print(f"Listening on {args.host}:{args.port}")
    print(f"Frame shape: {frame_shape}, dtype: {dtype}")

    run_server(
        camera_index=args.camera_index,
        frame_shape=frame_shape,
        dtype=dtype,
        port=args.port,
        host=args.host,
    )


if __name__ == "__main__":
    main()
