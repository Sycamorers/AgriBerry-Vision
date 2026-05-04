from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from detection_distribution.yolo_infer import resolve_infer_device

from .plant_detect import PlantDetection


@dataclass(frozen=True)
class MaskMeasurement:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    pixel_count: int
    mask: np.ndarray


class MobileSamSegmenter:
    def __init__(
        self,
        checkpoint: Path | str,
        device: str | int = "0",
        logger=None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found: {self.checkpoint}. "
                "Download mobile_sam.pt and place it under ckpt/ or pass --sam-checkpoint."
            )

        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "MobileSAM is not installed. Install it with "
                "`pip install git+https://github.com/ChaoningZhang/MobileSAM.git`."
            ) from exc

        self.device = resolve_infer_device(device=device, logger=logger)
        self.logger = logger
        model = sam_model_registry["vit_t"](checkpoint=str(self.checkpoint))
        model.to(device=self.device)
        model.eval()
        self.predictor = SamPredictor(model)

    def segment_boxes(
        self,
        image_bgr: np.ndarray,
        detections: List[PlantDetection],
    ) -> List[MaskMeasurement]:
        if image_bgr is None or image_bgr.size == 0 or not detections:
            return []

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        measurements: List[MaskMeasurement] = []
        for det in detections:
            box = np.asarray(det.xyxy, dtype=np.float32)
            try:
                masks, scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False,
                )
            except Exception:
                continue

            if masks is None or len(masks) == 0:
                continue

            if getattr(masks, "ndim", 0) == 3:
                best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
                mask = masks[best_idx]
                score = float(scores[best_idx]) if scores is not None and len(scores) > 0 else 0.0
            else:
                mask = masks
                score = float(scores[0]) if scores is not None and len(scores) > 0 else 0.0

            mask_bool = np.asarray(mask, dtype=bool)
            pixel_count = int(mask_bool.sum())
            if pixel_count <= 0:
                continue

            measurements.append(
                MaskMeasurement(
                    bbox_xyxy=det.xyxy,
                    score=score,
                    pixel_count=pixel_count,
                    mask=mask_bool,
                )
            )

        return measurements
