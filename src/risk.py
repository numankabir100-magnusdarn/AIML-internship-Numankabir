from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from .digital_twin import DigitalTwinEstimate, SimulatedPath


@dataclass(frozen=True)
class PredictedInteraction:
    track_a: int
    track_b: int
    risk_timeline: Mapping[float, float]
    per_horizon_risk: Mapping[float, float]
    min_distance_by_horizon: Mapping[float, float]
    time_to_threshold: float | None
    top_path_a: int
    top_path_b: int
    top_scenario_a: str
    top_scenario_b: str
    max_risk: float


@dataclass(frozen=True)
class RiskReport:
    interactions: Tuple[PredictedInteraction, ...] = ()
    pair_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskTrend:
    trend: str
    slope_per_second: float
    time_to_threshold: float | None
    current_risk: float


def compute_risk_trend(
    risk_timeline: Mapping[float, float],
    high_risk_threshold: float,
    per_horizon_risk: Mapping[float, float] | None = None,
) -> RiskTrend:
    if not risk_timeline:
        return RiskTrend("flat", 0.0, None, 0.0)

    horizons = np.asarray(sorted(risk_timeline), dtype=np.float64)
    risks = np.asarray([risk_timeline[float(h)] for h in horizons], dtype=np.float64)
    current_risk = float(risks[0])

    signal = per_horizon_risk if per_horizon_risk else risk_timeline
    signal_horizons = np.asarray(sorted(signal), dtype=np.float64)
    signal_values = np.asarray([signal[float(h)] for h in signal_horizons], dtype=np.float64)

    if len(signal_horizons) >= 2:
        slope = float(np.polyfit(signal_horizons, signal_values, 1)[0])
    else:
        slope = 0.0

    if slope >= 0.02:
        trend = "rising"
    elif slope <= -0.02:
        trend = "falling"
    else:
        trend = "flat"

    return RiskTrend(
        trend=trend,
        slope_per_second=slope,
        time_to_threshold=_time_to_threshold(horizons, risks, high_risk_threshold),
        current_risk=current_risk,
    )


def _time_to_threshold(
    horizons: np.ndarray,
    risks: np.ndarray,
    high_risk_threshold: float,
) -> Optional[float]:
    if high_risk_threshold <= 0.0 or len(risks) == 0:
        return None
    if float(risks[0]) >= high_risk_threshold:
        return float(horizons[0])

    for index in range(1, len(risks)):
        previous_risk = float(risks[index - 1])
        previous_time = float(horizons[index - 1])
        current_risk = float(risks[index])
        current_time = float(horizons[index])
        if current_risk >= high_risk_threshold:
            if current_risk <= previous_risk:
                return current_time
            fraction = (high_risk_threshold - previous_risk) / (current_risk - previous_risk)
            return previous_time + fraction * (current_time - previous_time)
    return None


class RiskEngine:
    def __init__(
        self,
        distance_threshold: float = 80.0,
        high_risk_threshold: float = 0.25,
    ) -> None:
        self.distance_threshold = max(1e-3, distance_threshold)
        self.high_risk_threshold = high_risk_threshold
        self._last_report: RiskReport | None = None

    def update(
        self,
        estimates: Mapping[int, DigitalTwinEstimate],
    ) -> RiskReport:
        if len(estimates) < 2:
            self._last_report = RiskReport()
            return self._last_report

        track_ids = sorted(estimates.keys())
        path_data = {
            track_id: self._path_arrays(estimate)
            for track_id, estimate in estimates.items()
        }

        interactions: List[PredictedInteraction] = []
        pair_scores: Dict[Tuple[int, int], float] = {}

        for index_a in range(len(track_ids)):
            for index_b in range(index_a + 1, len(track_ids)):
                track_a = track_ids[index_a]
                track_b = track_ids[index_b]
                positions_a, horizons_a, scores_a, scenarios_a = path_data[track_a]
                positions_b, horizons_b, scores_b, scenarios_b = path_data[track_b]

                interaction = self._evaluate_pair(
                    track_a,
                    track_b,
                    positions_a,
                    horizons_a,
                    scores_a,
                    scenarios_a,
                    positions_b,
                    horizons_b,
                    scores_b,
                    scenarios_b,
                )
                if interaction is not None:
                    interactions.append(interaction)
                pair_scores[(track_a, track_b)] = interaction.max_risk if interaction else 0.0

        report = RiskReport(
            interactions=tuple(sorted(interactions, key=lambda i: i.max_risk, reverse=True)),
            pair_scores=pair_scores,
        )
        self._last_report = report
        return report

    def high_risk_pairs(
        self,
        report: RiskReport | None = None,
    ) -> Set[Tuple[int, int]]:
        source = report if report is not None else self._last_report
        if source is None:
            return set()
        return {
            (interaction.track_a, interaction.track_b)
            for interaction in source.interactions
            if interaction.time_to_threshold is not None
            or interaction.max_risk >= self.high_risk_threshold
        }

    def high_risk_tracks(
        self,
        report: RiskReport | None = None,
    ) -> Set[int]:
        high_risk_pairs = self.high_risk_pairs(report)
        tracks: Set[int] = set()
        for track_a, track_b in high_risk_pairs:
            tracks.add(track_a)
            tracks.add(track_b)
        return tracks

    def _evaluate_pair(
        self,
        track_a: int,
        track_b: int,
        positions_a: np.ndarray,
        horizons_a: np.ndarray,
        scores_a: np.ndarray,
        scenarios_a: np.ndarray,
        positions_b: np.ndarray,
        horizons_b: np.ndarray,
        scores_b: np.ndarray,
        scenarios_b: np.ndarray,
    ) -> Optional[PredictedInteraction]:
        shared = min(positions_a.shape[1], positions_b.shape[1])
        if shared == 0:
            return None
        positions_a = positions_a[:, :shared, :]
        positions_b = positions_b[:, :shared, :]
        horizons = horizons_a[:shared]

        total_likelihood = float(scores_a.sum()) * float(scores_b.sum())
        if total_likelihood <= 1e-12:
            return None

        per_step_distance = np.linalg.norm(
            positions_a[:, None, :, :] - positions_b[None, :, :, :],
            axis=-1,
        )

        likelihood = scores_a[:, None] * scores_b[None, :]
        cumulative_distance = np.minimum.accumulate(per_step_distance, axis=-1)

        risk_timeline: Dict[float, float] = {}
        per_horizon_risk: Dict[float, float] = {}
        min_distance_by_horizon: Dict[float, float] = {}

        for horizon_index in range(shared):
            approach_distance = cumulative_distance[:, :, horizon_index]
            cumulative_risk = self._contact_risk(
                approach_distance, likelihood, total_likelihood
            )
            instantaneous_risk = self._contact_risk(
                per_step_distance[:, :, horizon_index], likelihood, total_likelihood
            )

            horizon = float(horizons[horizon_index])
            risk_timeline[horizon] = cumulative_risk
            per_horizon_risk[horizon] = instantaneous_risk
            min_distance_by_horizon[horizon] = float(np.min(per_step_distance[:, :, horizon_index]))

        if max(risk_timeline.values()) <= 0.0:
            return None

        risk_values = np.asarray(list(risk_timeline.values()), dtype=np.float64)
        time_to_threshold = _time_to_threshold(
            horizons,
            risk_values,
            self.high_risk_threshold,
        )
        max_risk = float(np.max(risk_values))

        best_horizon_index = int(np.argmax(risk_values))
        distance_at_best = per_step_distance[:, :, best_horizon_index]
        approach_at_best = distance_at_best < self.distance_threshold
        if bool(approach_at_best.any()):
            proximity_at_best = np.where(
                approach_at_best, 1.0 - distance_at_best / self.distance_threshold, 0.0
            )
            top_likelihood = likelihood * proximity_at_best
            top_flat = int(np.argmax(top_likelihood))
        else:
            top_flat = 0
        top_path_a, top_path_b = divmod(top_flat, distance_at_best.shape[1])

        return PredictedInteraction(
            track_a=track_a,
            track_b=track_b,
            risk_timeline=risk_timeline,
            per_horizon_risk=per_horizon_risk,
            min_distance_by_horizon=min_distance_by_horizon,
            time_to_threshold=time_to_threshold,
            top_path_a=int(top_path_a),
            top_path_b=int(top_path_b),
            top_scenario_a=str(scenarios_a[top_path_a]) if len(scenarios_a) else "normal",
            top_scenario_b=str(scenarios_b[top_path_b]) if len(scenarios_b) else "normal",
            max_risk=max_risk,
        )

    def _contact_risk(
        self,
        approach_distance: np.ndarray,
        likelihood: np.ndarray,
        total_likelihood: float,
    ) -> float:
        approach_mask = approach_distance < self.distance_threshold
        if not bool(approach_mask.any()):
            return 0.0
        proximity_weight = np.where(
            approach_mask, 1.0 - approach_distance / self.distance_threshold, 0.0
        )
        weighted_contact = likelihood * proximity_weight
        risk = float(weighted_contact.sum()) / total_likelihood
        return min(1.0, risk)

    def _path_arrays(
        self,
        estimate: DigitalTwinEstimate,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        paths: List[SimulatedPath] = list(estimate.simulated_paths)
        if not paths or not paths[0].checkpoints:
            empty = np.empty((0, 0, 3), dtype=np.float32)
            return empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=object)

        horizons = np.asarray(sorted(paths[0].checkpoints), dtype=np.float64)
        positions = np.array(
            [[path.checkpoints[float(h)] for h in horizons] for path in paths],
            dtype=np.float32,
        )
        scores = np.asarray([path.score for path in paths], dtype=np.float32)
        scenarios = np.asarray([path.scenario for path in paths], dtype=object)
        return positions, horizons, scores, scenarios