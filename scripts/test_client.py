"""Simple client to test the DCAM HTTP server."""
import requests
import numpy as np
import argparse
from pathlib import Path


def fetch_frame(url: str, x=None, y=None, w=None, h=None) -> tuple[np.ndarray, dict]:
    """Fetch frame or ROI from server.
    
    Args:
        url: Server base URL (e.g., 'http://localhost:8080')
        x, y, w, h: Optional ROI parameters
        
    Returns:
        (frame_array, metadata_dict)
    """
    params = {}
    if all(p is not None for p in [x, y, w, h]):
        params = {"x": x, "y": y, "w": w, "h": h}
    
    resp = requests.get(f"{url}/latest", params=params)
    resp.raise_for_status()
    
    # Extract metadata from headers
    metadata = {
        "frame_number": int(resp.headers["X-Frame-Number"]),
        "timestamp": float(resp.headers["X-Timestamp"]),
        "shape": tuple(map(int, resp.headers["X-Frame-Shape"].split(","))),
        "dtype": resp.headers["X-Frame-Dtype"]
    }
    
    # Reconstruct array
    dtype = np.dtype(metadata["dtype"])
    shape = metadata["shape"]
    frame = np.frombuffer(resp.content, dtype=dtype).reshape(shape)
    
    return frame, metadata


def main():
    parser = argparse.ArgumentParser(description="Test DCAM HTTP server")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--save", type=Path, help="Save frame to .npy file")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                       help="Fetch ROI instead of full frame")
    parser.add_argument("--show", action="store_true", help="Display frame with matplotlib")
    parser.add_argument("--vmin", type=float, help="Minimum value for display scaling")
    parser.add_argument("--vmax", type=float, help="Maximum value for display scaling")
    
    args = parser.parse_args()
    
    if args.roi:
        x, y, w, h = args.roi
        print(f"Fetching ROI: x={x}, y={y}, w={w}, h={h}")
        frame, metadata = fetch_frame(args.url, x, y, w, h)
    else:
        print("Fetching full frame")
        frame, metadata = fetch_frame(args.url)
    
    print(f"Frame #{metadata['frame_number']}")
    print(f"Timestamp: {metadata['timestamp']:.6f}")
    print(f"Shape: {frame.shape}")
    print(f"Dtype: {frame.dtype}")
    print(f"Min/Max: {frame.min()}/{frame.max()}")
    print(f"Mean: {frame.mean():.2f}")
    
    if args.save:
        np.save(args.save.with_suffix('.npy'), frame)
        print(f"Saved to {args.save.with_suffix('.npy')}")

        import matplotlib.pyplot as plt
        
        vmin = args.vmin if args.vmin is not None else frame.min()
        vmax = args.vmax if args.vmax is not None else frame.max()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Intensity')
        plt.title(f"Frame #{metadata['frame_number']} @ {metadata['timestamp']:.3f}s")
        plt.tight_layout()
        plt.savefig(args.save.with_suffix('.png'))
        print(f"Saved visualization to {args.save.with_suffix('.png')}")
    
    if args.show:
        import matplotlib.pyplot as plt
        
        vmin = args.vmin if args.vmin is not None else frame.min()
        vmax = args.vmax if args.vmax is not None else frame.max()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Intensity')
        plt.title(f"Frame #{metadata['frame_number']} @ {metadata['timestamp']:.3f}s")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()