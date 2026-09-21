from collections import defaultdict, deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, Dict, Iterable, Tuple

from .depth import SpatialTrack


@dataclass(frozen=True)
class VelocityEstimate:
    vx: float
    vy: float
    vz: float
    speed: float
    history_length: int


class VelocityEstimator:
    def __init__(self, history_size: int = 5) -> None:
        self._history_size = history_size
        self._tracks: Dict[int, Deque[Tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=self._history_size)
        )

    def update(self, spatial_tracks: Iterable[SpatialTrack]) -> Dict[int, VelocityEstimate]:
        active_track_ids = set()
        estimates: Dict[int, VelocityEstimate] = {}

        for spatial_track in spatial_tracks:
            active_track_ids.add(spatial_track.track_id)
            history = self._tracks[spatial_track.track_id]
            history.append(spatial_track.position_3d)
            estimates[spatial_track.track_id] = self._estimate(history)

        self.prune(active_track_ids)
        return estimates

    def prune(self, active_track_ids: Iterable[int]) -> None:
        active_ids = set(active_track_ids)
        for track_id in list(self._tracks.keys()):
            if track_id not in active_ids:
                del self._tracks[track_id]

    def _estimate(self, history: Deque[Tuple[float, float, float]]) -> VelocityEstimate:
        if len(history) < 2:
            return VelocityEstimate(0.0, 0.0, 0.0, 0.0, len(history))

        deltas = [
            (current_x - previous_x, current_y - previous_y, current_z - previous_z)
            for (previous_x, previous_y, previous_z), (current_x, current_y, current_z) in zip(history, list(history)[1:])
        ]
        vx = sum(delta_x for delta_x, _, _ in deltas) / len(deltas)
        vy = sum(delta_y for _, delta_y, _ in deltas) / len(deltas)
        vz = sum(delta_z for _, _, delta_z in deltas) / len(deltas)
        speed = sqrt(vx * vx + vy * vy + vz * vz)
        return VelocityEstimate(vx=vx, vy=vy, vz=vz, speed=speed, history_length=len(history))
