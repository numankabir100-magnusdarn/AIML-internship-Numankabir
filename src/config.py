from dataclasses import dataclass, field
from typing import Tuple


WEBCAM_INDEX = 0
OBS_VIRTUAL_CAM_INDEX = 2

# Default camera used when the app starts.
CAMERA_INDEX = WEBCAM_INDEX

# Named camera sources. The order here defines the cycle order used by the
# "C" hotkey at runtime; indexes may be integers (device id) or strings
# (video files / IP cameras / OS-specific device names).
CAMERA_SOURCES: dict[str, int] = {
    "Webcam": WEBCAM_INDEX,
    "OBS Virtual Camera": OBS_VIRTUAL_CAM_INDEX,
}


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = CAMERA_INDEX
    camera_source_name: str = "Webcam"
    model_path: str = "yolov8n.pt"
    depth_model_name: str = "MiDaS_small"
    digital_twin_simulations: int = 30
    digital_twin_duration_seconds: float = 5.0
    digital_twin_step_seconds: float = 0.1
    digital_twin_checkpoint_times: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    digital_twin_bands: Tuple[Tuple[float, float], ...] = (
        (0.1, 1.0), (0.25, 3.0), (0.5, 5.0)
    )
    digital_twin_fine_horizon_seconds: float = 1.0
    digital_twin_adaptive_count: bool = True
    digital_twin_near_risk_threshold: float = 0.05
    digital_twin_scenario_injection_rate: float = 0.2
    digital_twin_scenario_types: Tuple[str, ...] = ("stop", "turn", "speedup")
    risk_distance_threshold: float = 80.0
    risk_high_threshold: float = 0.25
    decision_monitor_risk: float = 0.15
    decision_caution_risk: float = 0.4
    decision_alert_risk: float = 0.7
    decision_alert_within_seconds: float = 1.0
    decision_caution_within_seconds: float = 2.0
    decision_monitor_within_seconds: float = 3.0
    confidence_threshold: float = 0.35
    image_size: int = 640
    velocity_history_size: int = 5
    window_name: str = "Predictive Spatial Intelligence - Phase 6"
