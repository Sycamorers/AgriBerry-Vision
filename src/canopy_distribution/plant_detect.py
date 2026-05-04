from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from detection_distribution.yolo_infer import get_model_class_names, resolve_infer_device


@dataclass(frozen=True)
class PlantDetection:
    class_id: int
    class_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


def load_image_bgr(image_path: Path | str) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return np.ascontiguousarray(image)


class PlantBoxDetector:
    def __init__(
        self,
        weights: Path | str,
        device: str | int = "0",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        logger=None,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.exists():
            raise FileNotFoundError(f"Plant detector weights not found: {self.weights}")

        self.device = resolve_infer_device(device=device, logger=logger)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.logger = logger

        self.model = YOLO(str(self.weights))
        self.class_names = get_model_class_names(self.model.names)

    def predict_image(self, image_bgr: np.ndarray) -> List[PlantDetection]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
        conf_arr = boxes.conf.detach().cpu().numpy().astype(float)
        xyxy_arr = boxes.xyxy.detach().cpu().numpy().astype(float)

        detections: List[PlantDetection] = []
        for idx in range(len(cls_arr)):
            class_id = int(cls_arr[idx])
            class_name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            x1, y1, x2, y2 = [float(v) for v in xyxy_arr[idx].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                PlantDetection(
                    class_id=class_id,
                    class_name=class_name,
                    conf=float(conf_arr[idx]),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections
