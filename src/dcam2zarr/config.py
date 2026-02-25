from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class Chunking(BaseModel):
    """Chunk size in pixels for each dimension."""
    x: Optional[int] = Field(None, gt=0, description="X chunk size (None = full dimension)")
    y: Optional[int] = Field(None, gt=0, description="Y chunk size (None = full dimension)")
    t: int = Field(64, gt=0, description="Time chunk size in frames")


class Sharding(BaseModel):
    """Number of chunks per shard for each dimension."""
    x: int = Field(1, gt=0, description="X sharding (1 = no sharding)")
    y: int = Field(1, gt=0, description="Y sharding (1 = no sharding)")
    t: int = Field(1, gt=0, description="Time sharding (1 = no sharding)")


class Monitor(BaseModel):
    """Live preview monitoring configuration."""
    enabled: bool = Field(False, description="Enable monitor stream")
    decimation: int = Field(10, gt=0, description="Publish every Nth frame")
    downsample_factor: int = Field(4, gt=0, description="Spatial downsampling factor")
    zenoh_topic: Optional[str] = Field(None, description="Zenoh topic (None = auto-generate)")

    @field_validator('zenoh_topic')
    @classmethod
    def validate_topic(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and ('/' not in v or v.startswith('/')):
            raise ValueError("zenoh_topic must be a valid topic path (e.g., 'camera/0/preview')")
        return v


class DCAMConfig(BaseModel):
    """DCAM to Zarr streaming configuration."""

    camera_index: int = Field(..., ge=0, description="Camera device index")
    output_path: Path = Field(..., description="Output Zarr store path")
    max_frames: Optional[int] = Field(None, gt=0, description="Max frames to capture (None = unlimited)")
    chunking: Chunking = Field(default_factory=Chunking)
    sharding: Sharding = Field(default_factory=Sharding)
    monitor: Monitor = Field(default_factory=Monitor)

    @classmethod
    def from_json_file(cls, path: Path) -> "DCAMConfig":
        """Load config from JSON file."""
        import json
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def to_json_file(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))