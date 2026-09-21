from dataclasses import dataclass
from typing import List, Tuple

from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass(frozen=True)
class Detection:
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def load_detector(model_path: str) -> YOLO:
    return YOLO(model_path)


def detect_frame(
    model: YOLO,
    frame,
    confidence_threshold: float,
    image_size: int,
) -> tuple[Results, List[Detection]]:
    results = model.predict(frame, conf=confidence_threshold, imgsz=image_size, verbose=False)
    primary_result = results[0]
    parsed_detections = _parse_detections(primary_result)
    return primary_result, parsed_detections


def _parse_detections(result: Results) -> List[Detection]:
    detections: List[Detection] = []
    if result.boxes is None:
        return detections

    boxes = result.boxes
    class_names = result.names

    for box in boxes:
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
        class_id = int(box.cls[0].item()) if box.cls is not None else -1
        class_name = class_names.get(class_id, str(class_id))
        detections.append(
            Detection(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
        )

    return detections
