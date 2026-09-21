import time

import cv2

from .camera import CameraSourceError, CameraSourceSwitcher
from .config import AppConfig
from .decision import DecisionEngine
from .detection import detect_frame, load_detector
from .depth import MiDaSDepthEstimator
from .digital_twin import DigitalTwinEngine
from .risk import RiskEngine
from .stats import SessionStats
from .tracking import ByteTrackTracker
from .velocity import VelocityEstimator
from .visualization import draw_annotations, draw_source_badge, draw_stats_panel


class PhaseOnePipeline:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.detector = load_detector(self.config.model_path)
        self.depth_estimator = MiDaSDepthEstimator(self.config.depth_model_name)
        self.tracker = ByteTrackTracker()
        self.velocity_estimator = VelocityEstimator(self.config.velocity_history_size)
        self.digital_twin_engine = DigitalTwinEngine(
            simulation_count=self.config.digital_twin_simulations,
            duration_seconds=self.config.digital_twin_duration_seconds,
            step_seconds=self.config.digital_twin_step_seconds,
            scenario_injection_rate=self.config.digital_twin_scenario_injection_rate,
            scenario_types=self.config.digital_twin_scenario_types,
            checkpoint_times=self.config.digital_twin_checkpoint_times,
            bands=self.config.digital_twin_bands,
            fine_horizon_seconds=self.config.digital_twin_fine_horizon_seconds,
            adaptive_count=self.config.digital_twin_adaptive_count,
            near_risk_threshold=self.config.digital_twin_near_risk_threshold,
            contact_distance_threshold=self.config.risk_distance_threshold,
        )
        self.risk_engine = RiskEngine(
            distance_threshold=self.config.risk_distance_threshold,
            high_risk_threshold=self.config.risk_high_threshold,
        )
        self.decision_engine = DecisionEngine(
            high_risk_threshold=self.config.risk_high_threshold,
            monitor_risk=self.config.decision_monitor_risk,
            caution_risk=self.config.decision_caution_risk,
            alert_risk=self.config.decision_alert_risk,
            alert_within_seconds=self.config.decision_alert_within_seconds,
            caution_within_seconds=self.config.decision_caution_within_seconds,
            monitor_within_seconds=self.config.decision_monitor_within_seconds,
        )
        self.stats = SessionStats()

    def run(self) -> None:
        switcher = CameraSourceSwitcher(
            initial_source=self.config.camera_source_name,
            initial_index=self.config.camera_index,
        )
        if not switcher.is_opened:
            raise RuntimeError(f"Unable to open webcam index {switcher.index}.")

        try:
            previous_frame_time = time.perf_counter()
            frame_count = 0
            warmup_frames = 3
            consecutive_read_failures = 0
            switch_error = ""
            while True:
                current_frame_time = time.perf_counter()
                frame_dt_seconds = current_frame_time - previous_frame_time
                previous_frame_time = current_frame_time

                success, frame = switcher.read()
                if not success:
                    # Allow a freshly-switched source a brief warm-up window
                    # before treating a failed read as fatal.
                    consecutive_read_failures += 1
                    if consecutive_read_failures > 5:
                        break
                    cv2.waitKey(33)
                    continue
                consecutive_read_failures = 0

                result, _detections = detect_frame(
                    self.detector,
                    frame,
                    self.config.confidence_threshold,
                    self.config.image_size,
                )
                tracked_objects = self.tracker.update(result)
                depth_map = self.depth_estimator.estimate_depth_map(frame)
                spatial_tracks = self.depth_estimator.project_tracks(tracked_objects, depth_map)
                velocity_estimates = self.velocity_estimator.update(spatial_tracks)
                digital_twin_estimates = self.digital_twin_engine.update(spatial_tracks, velocity_estimates, frame_dt_seconds)
                for spatial_track in spatial_tracks:
                    estimate = digital_twin_estimates.get(spatial_track.track_id)
                    path_count = len(estimate.simulated_paths) if estimate is not None else 0
                    print(f"[digital_twin] track={spatial_track.track_id} paths={path_count}")
                risk_report = self.risk_engine.update(digital_twin_estimates)
                decisions = self.decision_engine.update(risk_report)
                annotated_frame = draw_annotations(
                    frame,
                    spatial_tracks,
                    velocity_estimates,
                    digital_twin_estimates,
                    decisions=decisions,
                )

                self.stats.update(
                    track_ids=(tracked_object.track_id for tracked_object in tracked_objects),
                    risk_pairs_evaluated=len(risk_report.pair_scores),
                    decisions=decisions,
                )

                badge_lines = [
                    f"Source: {switcher.name}  |  [C] switch, [R] reset stats, [Q] quit",
                ]
                if switch_error:
                    badge_lines.append(switch_error)
                draw_source_badge(annotated_frame, badge_lines)
                draw_stats_panel(annotated_frame, self.stats.panel_lines())

                cv2.imshow(self.config.window_name, annotated_frame)
                frame_count += 1
                # The heavy inference loop starves OpenCV's GUI message pump, so on
                # Windows the freshly-created window can stay unshown/unpainted while
                # getWindowProperty reports visible. Pump longer than 1 ms for the
                # first few frames to let HighGUI finish window creation + first paint.
                delay_ms = 50 if (frame_count <= 3 or warmup_frames > 0) else 1
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("c"):
                    try:
                        new_name = switcher.switch_next()
                    except CameraSourceError as exc:
                        switch_error = f"Source '{switcher.name}': {exc}"
                    else:
                        switch_error = ""
                        warmup_frames = 3
                        print(f"[camera] switched to source '{new_name}' (index {switcher.index}).")
                elif key == ord("r"):
                    self.stats.reset()
                    print("[stats] session stats reset.")
                if warmup_frames > 0:
                    warmup_frames -= 1
        finally:
            switcher.release()
            cv2.destroyAllWindows()
