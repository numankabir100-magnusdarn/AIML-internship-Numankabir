from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .depth import SpatialTrack


@dataclass(frozen=True)
class PredictionEstimate:
    current_position_3d: Tuple[float, float, float]
    predicted_0_5s: Tuple[float, float, float]
    predicted_1_0s: Tuple[float, float, float]


class KalmanFilter3D:
    def __init__(self, measurement: Tuple[float, float, float]) -> None:
        self.state = np.array([
            measurement[0],
            measurement[1],
            measurement[2],
            0.0,
            0.0,
            0.0,
        ], dtype=np.float32).reshape(6, 1)
        self.covariance = np.eye(6, dtype=np.float32) * 500.0
        self.measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.measurement_noise = np.diag([12.0, 12.0, 0.03]).astype(np.float32)

    def predict(self, dt_seconds: float) -> None:
        transition = self._transition_matrix(dt_seconds)
        process_noise = self._process_noise(dt_seconds)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def update(self, measurement: Tuple[float, float, float]) -> None:
        measurement_vector = np.array(measurement, dtype=np.float32).reshape(3, 1)
        innovation = measurement_vector - self.measurement_matrix @ self.state
        innovation_covariance = (
            self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + self.measurement_noise
        )
        kalman_gain = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(6, dtype=np.float32)
        self.covariance = (identity - kalman_gain @ self.measurement_matrix) @ self.covariance

    def predict_position(self, horizon_seconds: float) -> Tuple[float, float, float]:
        transition = self._transition_matrix(horizon_seconds)
        predicted_state = transition @ self.state
        return float(predicted_state[0, 0]), float(predicted_state[1, 0]), float(predicted_state[2, 0])

    @staticmethod
    def _transition_matrix(dt_seconds: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, dt_seconds, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt_seconds, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt_seconds],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _process_noise(dt_seconds: float) -> np.ndarray:
        acceleration_variance = 2.5
        dt2 = dt_seconds * dt_seconds
        dt3 = dt2 * dt_seconds
        dt4 = dt2 * dt2
        one_axis = np.array(
            [
                [dt4 / 4.0, dt3 / 2.0],
                [dt3 / 2.0, dt2],
            ],
            dtype=np.float32,
        ) * acceleration_variance

        process_noise = np.zeros((6, 6), dtype=np.float32)
        process_noise[0:2, 0:2] = one_axis
        process_noise[2:4, 2:4] = one_axis
        process_noise[4:6, 4:6] = one_axis
        return process_noise


class PredictionEngine:
    def __init__(self, default_frame_dt: float = 1.0 / 30.0) -> None:
        self._default_frame_dt = default_frame_dt
        self._filters: Dict[int, KalmanFilter3D] = {}

    def update(
        self,
        spatial_tracks: Iterable[SpatialTrack],
        frame_dt_seconds: float | None,
    ) -> Dict[int, PredictionEstimate]:
        predictions: Dict[int, PredictionEstimate] = {}
        active_track_ids = set()
        update_dt = frame_dt_seconds if frame_dt_seconds and frame_dt_seconds > 0.0 else self._default_frame_dt

        for spatial_track in spatial_tracks:
            active_track_ids.add(spatial_track.track_id)
            current_position = spatial_track.position_3d
            kalman_filter = self._filters.get(spatial_track.track_id)

            if kalman_filter is None:
                kalman_filter = KalmanFilter3D(current_position)
                self._filters[spatial_track.track_id] = kalman_filter
            else:
                kalman_filter.predict(update_dt)
                kalman_filter.update(current_position)

            predictions[spatial_track.track_id] = PredictionEstimate(
                current_position_3d=current_position,
                predicted_0_5s=kalman_filter.predict_position(0.5),
                predicted_1_0s=kalman_filter.predict_position(1.0),
            )

        self.prune(active_track_ids)
        return predictions

    def prune(self, active_track_ids: Iterable[int]) -> None:
        active_ids = set(active_track_ids)
        for track_id in list(self._filters.keys()):
            if track_id not in active_ids:
                del self._filters[track_id]