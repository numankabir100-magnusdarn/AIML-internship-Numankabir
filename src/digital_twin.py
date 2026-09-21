from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .depth import SpatialTrack
from .velocity import VelocityEstimate


@dataclass(frozen=True)
class SimulatedPath:
    points: List[Tuple[float, float, float]]
    final_position: Tuple[float, float, float]
    score: float
    scenario: str = "normal"
    checkpoints: Mapping[float, Tuple[float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DigitalTwinEstimate:
    current_position_3d: Tuple[float, float, float]
    simulated_paths: List[SimulatedPath]


@dataclass
class TwinTrackState:
    last_position_3d: Tuple[float, float, float]
    last_velocity_3d: Tuple[float, float, float]
    seed: int


@dataclass
class _SimContext:
    rng: np.random.Generator
    times: np.ndarray
    dts: np.ndarray
    near_count: int
    points: np.ndarray
    velocity: np.ndarray
    applied: np.ndarray
    stopped: np.ndarray
    scenario_kind: np.ndarray
    scenario_step: np.ndarray
    is_turn: np.ndarray
    is_speedup: np.ndarray
    is_stop: np.ndarray
    turn_angle: np.ndarray
    speedup_boost: np.ndarray
    observed_speed: float
    start_position: Tuple[float, float, float]
    velocity0: np.ndarray


class DigitalTwinEngine:
    def __init__(
        self,
        simulation_count: int = 30,
        duration_seconds: float = 5.0,
        step_seconds: float = 0.1,
        direction_jitter: float = 0.18,
        speed_jitter: float = 0.12,
        scenario_injection_rate: float = 0.2,
        scenario_types: Sequence[str] = ("stop", "turn", "speedup"),
        checkpoint_times: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0),
        depth_jitter_scale: float = 0.3,
        bands: Sequence[Tuple[float, float]] = ((0.1, 1.0), (0.25, 3.0), (0.5, 5.0)),
        fine_horizon_seconds: float = 1.0,
        adaptive_count: bool = True,
        near_risk_threshold: float = 0.05,
        contact_distance_threshold: float = 80.0,
    ) -> None:
        self.simulation_count = max(1, simulation_count)
        self.duration_seconds = max(0.02, duration_seconds)
        self.step_seconds = max(0.02, step_seconds)
        self.direction_jitter = direction_jitter
        self.speed_jitter = speed_jitter
        self.depth_jitter_scale = min(1.0, max(0.0, depth_jitter_scale))
        self.scenario_injection_rate = min(1.0, max(0.0, scenario_injection_rate))
        self.scenario_types: Tuple[str, ...] = tuple(
            base for base in scenario_types
        )
        for scenario in ("stop", "turn", "speedup"):
            if scenario not in self.scenario_types:
                self.scenario_types = self.scenario_types + (scenario,)
        self._scenario_index = {
            name: index for index, name in enumerate(self.scenario_types)
        }
        self.checkpoint_times: Tuple[float, ...] = tuple(
            sorted(
                {
                    round(time, 6)
                    for time in checkpoint_times
                    if 0.0 < time <= self.duration_seconds
                }
            )
        )
        self.fine_horizon_seconds = min(self.duration_seconds, max(0.0, fine_horizon_seconds))
        self.adaptive_count = bool(adaptive_count)
        self.near_risk_threshold = max(0.0, near_risk_threshold)
        self.contact_distance_threshold = max(1e-3, contact_distance_threshold)

        cleaned = tuple(
            (dt, end)
            for dt, end in bands
            if dt > 0.0 and end > 0.0
        )
        if not cleaned:
            cleaned = ((self.step_seconds, self.duration_seconds),)
        self.bands: Tuple[Tuple[float, float], ...] = tuple(
            sorted(cleaned, key=lambda band: band[1])
        )
        if self.bands[-1][1] < self.duration_seconds:
            self.bands = self.bands + ((self.step_seconds, self.duration_seconds),)

        self._states: Dict[int, TwinTrackState] = {}
        self._frame_index = 0

    def update(
        self,
        spatial_tracks: Iterable[SpatialTrack],
        velocity_estimates: Dict[int, VelocityEstimate],
        frame_dt_seconds: float | None,
    ) -> Dict[int, DigitalTwinEstimate]:
        estimates: Dict[int, DigitalTwinEstimate] = {}
        active_track_ids = set()
        self._frame_index += 1
        time_scale = frame_dt_seconds if frame_dt_seconds and frame_dt_seconds > 0.0 else 1.0 / 30.0

        contexts: Dict[int, _SimContext] = {}
        for spatial_track in spatial_tracks:
            track_id = spatial_track.track_id
            active_track_ids.add(track_id)

            current_position = spatial_track.position_3d
            observed_velocity = self._velocity_vector(velocity_estimates.get(track_id), time_scale)
            state = self._states.get(track_id)

            if state is None:
                state = TwinTrackState(
                    last_position_3d=current_position,
                    last_velocity_3d=observed_velocity,
                    seed=(track_id * 1009) & 0xFFFFFFFF,
                )
                self._states[track_id] = state
            else:
                blended_velocity = tuple(
                    0.65 * previous + 0.35 * observed
                    for previous, observed in zip(state.last_velocity_3d, observed_velocity)
                )
                state.last_position_3d = current_position
                state.last_velocity_3d = blended_velocity

            contexts[track_id] = self._prepare_context(
                track_id,
                current_position,
                state.last_velocity_3d,
            )

        for ctx in contexts.values():
            self._propagate_near(ctx)

        if self.adaptive_count:
            detailed_ids = self._assess_near_term_activity(contexts)
        else:
            detailed_ids = set(contexts.keys())

        for track_id, ctx in contexts.items():
            self._extend_far(ctx, track_id in detailed_ids)
            paths = self._build_paths(ctx)
            estimates[track_id] = DigitalTwinEstimate(
                current_position_3d=ctx.start_position,
                simulated_paths=paths,
            )

        self.prune(active_track_ids)
        return estimates

    def prune(self, active_track_ids: Iterable[int]) -> None:
        active_ids = set(active_track_ids)
        for track_id in list(self._states.keys()):
            if track_id not in active_ids:
                del self._states[track_id]

    def _time_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        times: List[float] = [0.0]
        last = 0.0
        for band_dt, band_end in self.bands:
            if band_end <= last:
                continue
            while last < band_end - 1e-9:
                step = min(band_dt, band_end - last)
                last = round(last + step, 6)
                times.append(last)
        combined = sorted(set(times) | set(self.checkpoint_times))
        bounds = self.duration_seconds + 1e-9
        times_arr = np.asarray([t for t in combined if t <= bounds], dtype=np.float64)
        dts = np.diff(times_arr, prepend=0.0)
        return times_arr, dts

    def _prepare_context(
        self,
        track_id: int,
        start_position: Tuple[float, float, float],
        observed_velocity: Tuple[float, float, float],
    ) -> _SimContext:
        rng = np.random.default_rng(self._frame_index * 7919 + self._states[track_id].seed)
        times, dts = self._time_grid()
        total_steps = len(times)
        sims = self.simulation_count

        velocity0 = np.array(observed_velocity, dtype=np.float32)
        observed_speed = float(np.linalg.norm(velocity0))
        if observed_speed < 1e-4:
            velocity0 = rng.normal(0.0, 0.35, size=3).astype(np.float32)
            observed_speed = float(np.linalg.norm(velocity0))
            if observed_speed < 1e-4:
                velocity0 = np.array([0.0, 0.0, 0.05], dtype=np.float32)
                observed_speed = float(np.linalg.norm(velocity0))

        scenario_kind = np.full(sims, -1, dtype=np.int64)
        selected = np.flatnonzero(rng.random(sims) < self.scenario_injection_rate)
        if selected.size:
            scenario_kind[selected] = rng.integers(0, len(self.scenario_types), size=selected.size)
        scenario_step = np.zeros(sims, dtype=np.int64)
        if total_steps > 1:
            step_fraction = rng.uniform(0.35, 0.7, size=sims)
            scenario_step = np.minimum(
                total_steps - 1,
                np.maximum(1, np.round(step_fraction * (total_steps - 1)).astype(np.int64)),
            )

        turn_angle = (
            np.deg2rad(rng.uniform(45.0, 90.0, size=sims)) * rng.choice([-1.0, 1.0], size=sims)
        ).astype(np.float32)
        speedup_boost = rng.uniform(1.6, 2.6, size=sims).astype(np.float32)

        track_kind = scenario_kind
        is_turn = track_kind == self._scenario_index.get("turn", -1)
        is_speedup = track_kind == self._scenario_index.get("speedup", -1)
        is_stop = track_kind == self._scenario_index.get("stop", -1)

        near_indices = np.flatnonzero(times <= self.fine_horizon_seconds + 1e-9)
        near_count = int(near_indices[-1] + 1) if near_indices.size else 1

        points = np.empty((sims, total_steps, 3), dtype=np.float32)
        points[:, 0] = np.asarray(start_position, dtype=np.float32)
        velocity = np.tile(velocity0, (sims, 1))

        return _SimContext(
            rng=rng,
            times=times,
            dts=dts,
            near_count=near_count,
            points=points,
            velocity=velocity,
            applied=np.zeros(sims, dtype=bool),
            stopped=np.zeros(sims, dtype=bool),
            scenario_kind=scenario_kind,
            scenario_step=scenario_step,
            is_turn=is_turn,
            is_speedup=is_speedup,
            is_stop=is_stop,
            turn_angle=turn_angle,
            speedup_boost=speedup_boost,
            observed_speed=observed_speed,
            start_position=start_position,
            velocity0=velocity0,
        )

    def _assess_near_term_activity(self, contexts: Dict[int, _SimContext]) -> set:
        active: set = set()
        if len(contexts) < 2 or self.near_risk_threshold <= 0.0:
            return set(contexts.keys())

        track_ids = list(contexts.keys())
        threshold = self.contact_distance_threshold
        for index_a in range(len(track_ids)):
            for index_b in range(index_a + 1, len(track_ids)):
                ctx_a = contexts[track_ids[index_a]]
                ctx_b = contexts[track_ids[index_b]]
                near_a = ctx_a.points[:, : ctx_a.near_count, :]
                near_b = ctx_b.points[:, : ctx_b.near_count, :]

                end_a = near_a[:, -1, :]
                end_b = near_b[:, -1, :]
                centroid_a = near_a.reshape(-1, 3).mean(axis=0)
                centroid_b = near_b.reshape(-1, 3).mean(axis=0)
                spread_a = float(np.linalg.norm(near_a - centroid_a, axis=-1).max())
                spread_b = float(np.linalg.norm(near_b - centroid_b, axis=-1).max())
                if np.linalg.norm(centroid_a - centroid_b) > threshold + spread_a + spread_b:
                    continue

                distances = np.linalg.norm(
                    near_a[:, None, :, :] - near_b[None, :, :, :], axis=-1
                )
                closest = np.minimum.accumulate(distances, axis=-1)[:, :, -1]
                contact_fraction = float(np.mean(closest < threshold))
                if contact_fraction >= self.near_risk_threshold:
                    active.add(track_ids[index_a])
                    active.add(track_ids[index_b])
        return active

    def _propagate_near(self, ctx: _SimContext) -> None:
        if len(ctx.times) > 1 and ctx.near_count > 1:
            self._propagate(ctx, 1, ctx.near_count)

    def _extend_far(self, ctx: _SimContext, detailed: bool) -> None:
        total_steps = len(ctx.times)
        if ctx.near_count >= total_steps:
            return
        if detailed:
            self._propagate(ctx, ctx.near_count, total_steps)
        else:
            self._fill_extrapolated(ctx)

    def _propagate(
        self,
        ctx: _SimContext,
        index_start: int,
        index_end: int,
    ) -> None:
        rng = ctx.rng
        points = ctx.points
        velocity = ctx.velocity
        noise_axis = np.asarray([1.0, 1.0, self.depth_jitter_scale], dtype=np.float32)
        jitter_scale = max(ctx.observed_speed, 1.0)

        for abs_index in range(index_start, index_end):
            dt = float(ctx.dts[abs_index])
            trigger = (~ctx.applied) & (ctx.scenario_kind >= 0) & (abs_index >= ctx.scenario_step)
            if bool(trigger.any()):
                turn_mask = trigger & ctx.is_turn
                if bool(turn_mask.any()):
                    xy = velocity[turn_mask, :2]
                    angle = ctx.turn_angle[turn_mask]
                    cosine, sine = np.cos(angle), np.sin(angle)
                    velocity[turn_mask, 0] = xy[:, 0] * cosine - xy[:, 1] * sine
                    velocity[turn_mask, 1] = xy[:, 0] * sine + xy[:, 1] * cosine
                speedup_mask = trigger & ctx.is_speedup
                if bool(speedup_mask.any()):
                    velocity[speedup_mask] = velocity[speedup_mask] * ctx.speedup_boost[speedup_mask, None]
                ctx.stopped |= (trigger & ctx.is_stop)
                ctx.applied |= trigger

            moving = ~ctx.stopped
            if bool(moving.any()):
                count = int(moving.sum())
                speed_scale = 1.0 + rng.normal(0.0, self.speed_jitter, size=count)
                noise = rng.normal(0.0, self.direction_jitter, size=(count, 3)).astype(np.float32)
                noise *= noise_axis
                velocity[moving] = velocity[moving] * speed_scale[:, None] + noise * jitter_scale
                points[moving, abs_index] = points[moving, abs_index - 1] + velocity[moving] * dt
            points[~moving, abs_index] = points[~moving, abs_index - 1]

    def _fill_extrapolated(self, ctx: _SimContext) -> None:
        points = ctx.points
        pivot_index = ctx.near_count - 1
        pivot_time = float(ctx.times[pivot_index])
        anchor = points[:, pivot_index]
        velocity = ctx.velocity
        for abs_index in range(ctx.near_count, len(ctx.times)):
            elapsed = float(ctx.times[abs_index]) - pivot_time
            points[:, abs_index] = anchor + velocity * elapsed

    def _build_paths(self, ctx: _SimContext) -> List[SimulatedPath]:
        points = ctx.points
        sims = points.shape[0]
        start_vector = np.asarray(ctx.start_position, dtype=np.float32)
        endpoints = points[:, -1]
        path_velocity = (endpoints - start_vector) / max(self.duration_seconds, 1e-3)
        deviation = path_velocity - ctx.velocity0
        penalty = np.linalg.norm(deviation, axis=1)
        scores = np.exp(-penalty / max(ctx.observed_speed + 1e-3, 1.0))
        order = np.argsort(-scores)

        paths: List[SimulatedPath] = []
        for rank in order.tolist():
            series = points[rank]
            point_list = [
                (float(value[0]), float(value[1]), float(value[2]))
                for value in series
            ]
            final_position = point_list[-1]
            kind = int(ctx.scenario_kind[rank])
            scenario = self.scenario_types[kind] if kind >= 0 else "normal"
            paths.append(
                SimulatedPath(
                    points=point_list,
                    final_position=final_position,
                    score=float(scores[rank]),
                    scenario=scenario,
                    checkpoints=self._extract_checkpoints(series, ctx.times),
                )
            )
        return paths

    def _extract_checkpoints(
        self,
        points_1d: np.ndarray,
        times: np.ndarray,
    ) -> Dict[float, Tuple[float, float, float]]:
        if not self.checkpoint_times or points_1d.shape[0] < 2:
            return {}
        horizon_values = np.asarray(self.checkpoint_times, dtype=np.float64)
        indices = np.argmin(np.abs(times[:, None] - horizon_values[None, :]), axis=0)
        checkpoints: Dict[float, Tuple[float, float, float]] = {}
        for horizon, index in zip(self.checkpoint_times, indices.tolist()):
            point = points_1d[index]
            checkpoints[float(horizon)] = (
                float(point[0]), float(point[1]), float(point[2])
            )
        return checkpoints

    @staticmethod
    def _velocity_vector(
        velocity_estimate: VelocityEstimate | None,
        time_scale_seconds: float,
    ) -> Tuple[float, float, float]:
        if velocity_estimate is None:
            return 0.0, 0.0, 0.0

        safe_scale = max(time_scale_seconds, 1e-3)
        return (
            velocity_estimate.vx / safe_scale,
            velocity_estimate.vy / safe_scale,
            velocity_estimate.vz / safe_scale,
        )