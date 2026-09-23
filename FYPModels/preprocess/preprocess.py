from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class PreprocessSummary:
    source_video: str
    source_fps: float
    sampled_fps: float
    sampled_frames: int
    windows_written: int
    manifest_path: str | None
    diagnostics_path: str | None = None
    run_record_path: str | None = None