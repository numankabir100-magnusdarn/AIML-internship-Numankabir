from dataclasses import dataclass
from typing import List, Mapping, Tuple

import numpy as np
import supervision as sv
from ultralytics.engine.results import Results


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class ByteTrackTracker:
    def __init__(self) -> None:
        self._tracker = sv.ByteTrack()

    def update(self, result: Results) -> List[TrackedObject]:
        detections = sv.Detections.from_ultralytics(result)
        tracked_detections = self._tracker.update_with_detections(detections)
        return self._convert(tracked_detections, result.names)

    def _convert(
        self,
        detections: sv.Detections,
        class_names: Mapping[int, str],
    ) -> List[TrackedObject]:
        tracked_objects: List[TrackedObject] = []

        if len(detections) == 0 or detections.tracker_id is None:
            return tracked_objects

        confidences = detections.confidence
        class_ids = detections.class_id

        if confidences is None:
            confidences = np.zeros(len(detections), dtype=float)
        if class_ids is None:
            class_ids = np.full(len(detections), -1, dtype=int)

        for xyxy, confidence, class_id, track_id in zip(
            detections.xyxy,
            confidences,
            class_ids,
            detections.tracker_id,
        ):
            if track_id is None:
                continue

            class_id_int = int(class_id)
            tracked_objects.append(
                TrackedObject(
                    track_id=int(track_id),
                    xyxy=tuple(float(value) for value in xyxy.tolist()),
                    confidence=float(confidence),
                    class_id=class_id_int,
                    class_name=class_names.get(class_id_int, str(class_id_int)),
                )
            )

        return tracked_objects
