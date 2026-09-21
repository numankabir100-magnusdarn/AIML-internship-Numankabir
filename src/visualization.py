from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from .decision import Decision
from .digital_twin import DigitalTwinEstimate
from .depth import SpatialTrack
from .velocity import VelocityEstimate

_DECISION_STYLE = {
    "alert": ((0, 0, 255), 3),
    "caution": ((0, 165, 255), 2),
    "monitor": ((0, 255, 255), 2),
}


def draw_source_badge(frame, lines: list[str]) -> None:
    """Draw a small status badge (top-left) with each line on its own row."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    margin, pad_x, pad_y, line_spacing = 10, 8, 5, 2

    metrics = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    max_width = max((size[0] for size, _ in metrics), default=0)
    text_height = max((size[1] for size, _ in metrics), default=0)
    block_height = len(lines) * (text_height + line_spacing) - line_spacing + 2 * pad_y

    x0, y0 = margin, margin
    x1, y1 = x0 + max_width + 2 * pad_x, y0 + block_height
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), cv2.FILLED)

    text_x = x0 + pad_x
    for position, line in enumerate(lines):
        baseline = y0 + pad_y + text_height + position * (text_height + line_spacing)
        cv2.putText(frame, line, (text_x, baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_stats_panel(frame, lines: list[str], alpha: float = 0.45) -> None:
    """Draw a semi-transparent stats panel (top-right) with one row per line."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    margin, pad_x, pad_y, line_spacing = 10, 10, 8, 18

    metrics = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    max_width = max((size[0] for size, _ in metrics), default=0)
    panel_width = max_width + 2 * pad_x
    panel_height = len(lines) * line_spacing + 2 * pad_y

    frame_height, frame_width = frame.shape[:2]
    x0 = max(0, frame_width - panel_width - margin)
    y0 = margin
    x1 = min(frame_width, x0 + panel_width)
    y1 = y0 + panel_height

    panel = frame[y0:y1, x0:x1]
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], panel.shape[0]), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0.0, dst=panel)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (120, 120, 120), 1)

    for position, line in enumerate(lines):
        baseline = y0 + pad_y + line_spacing * position + metrics[position][1]
        color = (255, 255, 255)
        if position == 0:
            color = (80, 200, 255)
        cv2.putText(frame, line, (x0 + pad_x, baseline), font, scale, color, thickness, cv2.LINE_AA)


def draw_annotations(
    frame,
    spatial_tracks: list[SpatialTrack],
    velocity_estimates: Mapping[int, VelocityEstimate],
    digital_twin_estimates: Mapping[int, DigitalTwinEstimate],
    decisions: Optional[Mapping[int, Decision]] = None,
):
    annotated = frame.copy()
    twin_overlay = np.zeros_like(annotated)

    for spatial_track in spatial_tracks:
        x1, y1, x2, y2 = (int(value) for value in spatial_track.xyxy)
        color = _color_for_id(spatial_track.track_id)
        label = _build_label(spatial_track, velocity_estimates.get(spatial_track.track_id))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _draw_label(annotated, label, x1, y1, color)

        digital_twin_estimate = digital_twin_estimates.get(spatial_track.track_id)
        if digital_twin_estimate is not None:
            _draw_probability_cone(
                twin_overlay,
                spatial_track.centroid_2d,
                digital_twin_estimate,
                color,
            )

    result = cv2.addWeighted(annotated, 1.0, twin_overlay, 0.9, 0.0)

    if decisions is not None:
        for spatial_track in spatial_tracks:
            decision = decisions.get(spatial_track.track_id)
            if decision is not None and decision.action != "no_action":
                _draw_decision(result, spatial_track, decision)

    return result


def _draw_decision(frame, spatial_track: SpatialTrack, decision: Decision) -> None:
    style = _DECISION_STYLE.get(decision.action)
    if style is None:
        return
    color, thickness = style
    x1, y1, x2, y2 = (int(value) for value in spatial_track.xyxy)
    padding = 4 if decision.action == "alert" else 3
    cv2.rectangle(
        frame,
        (x1 - padding, y1 - padding),
        (x2 + padding, y2 + padding),
        color,
        thickness,
    )
    cv2.putText(
        frame,
        decision.action.upper(),
        (max(x1 - padding, 0), max(y1 - 14, 14)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _build_label(
    spatial_track: SpatialTrack,
    velocity_estimate: VelocityEstimate | None,
) -> str:
    label = (
        f"{spatial_track.class_name} #{spatial_track.track_id} "
        f"depth: {spatial_track.depth_relative:.2f} rel"
    )
    if velocity_estimate is not None:
        label += (
            f" | v3d=({velocity_estimate.vx:.1f}, {velocity_estimate.vy:.1f}, "
            f"{velocity_estimate.vz:.2f})"
        )
    return label


def _draw_label(frame, label: str, x: int, y: int, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(y - text_height - baseline - 6, 0)
    bottom = top + text_height + baseline + 6
    right = x + text_width + 8

    cv2.rectangle(frame, (x, top), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label, (x + 4, bottom - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_prediction_marker(
    frame,
    current_x: float,
    current_y: float,
    predicted_position: tuple[float, float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    predicted_x, predicted_y, _ = predicted_position
    current_point = (int(round(current_x)), int(round(current_y)))
    predicted_point = (int(round(predicted_x)), int(round(predicted_y)))

    cv2.line(frame, current_point, predicted_point, color, 1, cv2.LINE_AA)
    cv2.circle(frame, predicted_point, 5, color, cv2.FILLED, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (predicted_point[0] + 6, predicted_point[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_probability_cone(
    overlay,
    current_position_2d: tuple[float, float],
    estimate: DigitalTwinEstimate,
    base_color: tuple[int, int, int],
) -> None:
    current_point = (int(round(current_position_2d[0])), int(round(current_position_2d[1])))

    for path in estimate.simulated_paths:
        alpha = 0.18 + 0.82 * float(path.score)
        path_color = tuple(max(32, int(channel * alpha)) for channel in base_color)
        path_points = [(int(round(x)), int(round(y))) for x, y, _ in path.points]

        for start_point, end_point in zip(path_points, path_points[1:]):
            cv2.line(overlay, start_point, end_point, path_color, 2, cv2.LINE_AA)

        end_point = path_points[-1]
        cv2.circle(overlay, end_point, 3, path_color, cv2.FILLED, cv2.LINE_AA)
        cv2.line(overlay, current_point, end_point, tuple(max(24, int(channel * (alpha * 0.55))) for channel in base_color), 2, cv2.LINE_AA)


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id + 1337)
    color = rng.integers(64, 224, size=3)
    return int(color[0]), int(color[1]), int(color[2])
