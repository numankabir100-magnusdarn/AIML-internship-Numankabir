from time import perf_counter
from typing import Iterable, Mapping

from .decision import Decision


_ACTIVE_DECISION_ACTIONS = ("alert", "caution", "monitor")


class SessionStats:
    """Running totals for the live demo, observed purely from outputs other
    modules already produce. This class never modifies detection, tracking,
    depth, digital-twin, or risk logic.

    Decision events are counted per-episode: a new event is recorded only when
    a track's decision action *changes into* alert/caution/monitor from a
    different action. A track that stays in alert for 40 frames still counts
    as one alert event.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start_time = perf_counter()
        self._frame_count = 0
        self._unique_track_ids = set()
        self._active_track_ids = set()
        self._decision_episodes = {action: 0 for action in _ACTIVE_DECISION_ACTIONS}
        self._last_action_by_track = {}
        self._risk_pairs_evaluated = 0

    def update(
        self,
        track_ids: Iterable[int],
        risk_pairs_evaluated: int,
        decisions: Mapping[int, Decision],
    ) -> None:
        active_ids = [int(track_id) for track_id in track_ids]
        self._unique_track_ids.update(active_ids)
        self._active_track_ids = set(active_ids)

        for track_id, decision in decisions.items():
            action = decision.action
            previous = self._last_action_by_track.get(track_id, None)
            if action in _ACTIVE_DECISION_ACTIONS and action != previous:
                self._decision_episodes[action] += 1
            self._last_action_by_track[track_id] = action

        self._risk_pairs_evaluated += max(0, int(risk_pairs_evaluated))
        self._frame_count += 1

    @property
    def uptime_seconds(self) -> float:
        return perf_counter() - self._start_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def active_object_count(self) -> int:
        return len(self._active_track_ids)

    @property
    def total_unique_objects(self) -> int:
        return len(self._unique_track_ids)

    @property
    def alert_episodes(self) -> int:
        return self._decision_episodes["alert"]

    @property
    def caution_episodes(self) -> int:
        return self._decision_episodes["caution"]

    @property
    def monitor_episodes(self) -> int:
        return self._decision_episodes["monitor"]

    @property
    def risk_pairs_evaluated(self) -> int:
        return self._risk_pairs_evaluated

    @property
    def average_fps(self) -> float:
        elapsed = self.uptime_seconds
        if elapsed <= 0.0:
            return 0.0
        return self._frame_count / elapsed

    @property
    def active_decision_episodes(self) -> dict[str, int]:
        return dict(self._decision_episodes)

    def panel_lines(self) -> list[str]:
        uptime = int(self.uptime_seconds)
        return [
            "Session Stats",
            f"Uptime: {uptime // 3600:02d}:{(uptime % 3600) // 60:02d}:{uptime % 60:02d}",
            f"Active objects: {self.active_object_count}",
            f"Total detected: {self.total_unique_objects}",
            f"Alerts: {self.alert_episodes}",
            f"Cautions: {self.caution_episodes}",
            f"Monitors: {self.monitor_episodes}",
            f"Risk pairs: {self.risk_pairs_evaluated}",
            f"Avg FPS: {self.average_fps:.1f}",
        ]