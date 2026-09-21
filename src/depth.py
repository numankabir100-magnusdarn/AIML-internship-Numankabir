from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .tracking import TrackedObject


@dataclass(frozen=True)
class SpatialTrack:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    centroid_2d: Tuple[float, float]
    depth_relative: float
    position_3d: Tuple[float, float, float]


class MiDaSDepthEstimator:
    def __init__(self, model_name: str = "MiDaS_small") -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = self.transforms.small_transform if model_name == "MiDaS_small" else self.transforms.dpt_transform

    def estimate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(frame_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        depth_map = prediction.squeeze().cpu().numpy().astype(np.float32)
        return self._normalize(depth_map)

    def project_tracks(
        self,
        tracked_objects: Iterable[TrackedObject],
        depth_map: np.ndarray,
    ) -> List[SpatialTrack]:
        spatial_tracks: List[SpatialTrack] = []
        height, width = depth_map.shape[:2]

        for tracked_object in tracked_objects:
            centroid_x, centroid_y = self._centroid(tracked_object.xyxy)
            center_x = self._clamp_index(centroid_x, width)
            center_y = self._clamp_index(centroid_y, height)
            depth_relative = float(depth_map[center_y, center_x])
            position_3d = (centroid_x, centroid_y, depth_relative)

            spatial_tracks.append(
                SpatialTrack(
                    track_id=tracked_object.track_id,
                    xyxy=tracked_object.xyxy,
                    confidence=tracked_object.confidence,
                    class_id=tracked_object.class_id,
                    class_name=tracked_object.class_name,
                    centroid_2d=(centroid_x, centroid_y),
                    depth_relative=depth_relative,
                    position_3d=position_3d,
                )
            )

        return spatial_tracks

    def _load_model(self):
        model = torch.hub.load("intel-isl/MiDaS", self.model_name, trust_repo=True)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _normalize(depth_map: np.ndarray) -> np.ndarray:
        minimum = float(np.min(depth_map))
        maximum = float(np.max(depth_map))
        if maximum - minimum < 1e-6:
            return np.zeros_like(depth_map, dtype=np.float32)
        normalized = (depth_map - minimum) / (maximum - minimum)
        return normalized.astype(np.float32)

    @staticmethod
    def _centroid(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _clamp_index(value: float, upper_bound: int) -> int:
        return max(0, min(int(round(value)), upper_bound - 1))