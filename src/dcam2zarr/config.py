from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CompressionCodec(str, Enum):
    """Compression codec options."""

    ZSTD = "zstd"
    LZ4 = "lz4"


class DownsamplingMethod(str, Enum):
    """Downsampling method for multiscale."""

    MEAN = "mean"
    MIN = "min"
    MAX = "max"


class Compression(BaseModel):
    """Compression configuration."""

    enabled: bool = Field(False, description="Enable compression")
    codec: CompressionCodec = Field(
        CompressionCodec.ZSTD, description="Compression codec"
    )
    level: int = Field(3, ge=0, le=9, description="Compression level (0-9)")


class Multiscale(BaseModel):
    """Multiscale pyramid configuration."""

    enabled: bool = Field(False, description="Enable multiscale pyramids")
    method: DownsamplingMethod = Field(
        DownsamplingMethod.MEAN, description="Downsampling method"
    )


class Chunking(BaseModel):
    """Chunk size in pixels for each dimension."""

    x: Optional[int] = Field(
        None, gt=0, description="X chunk size (None = full dimension)"
    )
    y: Optional[int] = Field(
        None, gt=0, description="Y chunk size (None = full dimension)"
    )
    t: int = Field(64, gt=0, description="Time chunk size in frames")


class Sharding(BaseModel):
    """Number of chunks per shard for each dimension."""

    x: int = Field(1, gt=0, description="X sharding (1 = no sharding)")
    y: int = Field(1, gt=0, description="Y sharding (1 = no sharding)")
    t: int = Field(1, gt=0, description="Time sharding (1 = no sharding)")


class HttpServer(BaseModel):
    """HTTP server configuration."""

    enabled: bool = Field(False, description="Enable HTTP server")
    host: str = Field("0.0.0.0", description="Host address to bind to")
    port: int = Field(8080, gt=0, description="Port to bind to")


class DCAMConfig(BaseModel):
    """DCAM to Zarr streaming configuration."""

    camera_index: int = Field(..., ge=0, description="Camera device index")
    output_path: Path = Field(..., description="Output Zarr store path")
    max_frames: Optional[int] = Field(
        None, gt=0, description="Max frames to capture (None = unlimited)"
    )
    chunking: Chunking = Field(default_factory=Chunking)
    sharding: Sharding = Field(default_factory=Sharding)
    compression: Compression = Field(default_factory=Compression)
    multiscale: Multiscale = Field(default_factory=Multiscale)
    http_server: HttpServer = Field(default_factory=HttpServer)

    @classmethod
    def from_json_file(cls, path: Path) -> "DCAMConfig":
        """Load config from JSON file."""
        import json

        with open(path) as f:
            return cls.model_validate(json.load(f))

    def to_json_file(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
