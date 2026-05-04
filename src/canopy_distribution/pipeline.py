from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd

from detection_distribution.aggregate import aggregate_points
from detection_distribution.gps_parse import parse_image_path
from detection_distribution.io_scan import SequenceInfo, collect_image_files
from detection_distribution.qgis_export import (
    write_distribution_layers_from_agg_df,
    write_qgis_points_from_agg_df,
)
from detection_distribution.utils import ensure_dir

from .mobile_sam import MaskMeasurement, MobileSamSegmenter
from .plant_detect import PlantBoxDetector, load_image_bgr


CANOPY_METRIC_NAME = "canopy_size"


@dataclass(frozen=True)
class TrialProcessResult:
    index_rows: List[Dict[str, Any]]
    n_images_total: int
    n_images_parsed_ok: int
    n_parse_fail: int
    n_gps_points: int
    n_detections_total: int
    n_masks_total: int


def _parse_sequence_images(
    seq_info: SequenceInfo,
    image_extensions: Sequence[str],
    logger,
) -> tuple[List[Dict[str, Any]], int, int]:
    image_paths = collect_image_files(seq_info.seq_dir, image_extensions=image_extensions)
    parsed_records: List[Dict[str, Any]] = []
    parse_fail_count = 0

    for image_path in image_paths:
        parsed = parse_image_path(image_path)
        if not parsed.parse_ok:
            parse_fail_count += 1
            logger.warning("GPS parse failed: %s | reason=%s", image_path, parsed.error)
            continue

        parsed_records.append(
            {
                "box": seq_info.box,
                "seq": seq_info.seq,
                "image_path": str(image_path),
                "timestamp": parsed.timestamp,
                "lat": float(parsed.lat),
                "lon": float(parsed.lon),
                "parse_ok": True,
            }
        )

    return parsed_records, len(image_paths), parse_fail_count


def _ensure_per_image_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    for base_col, default in [
        ("box", ""),
        ("seq", ""),
        ("image_path", ""),
        ("timestamp", ""),
        ("lat", np.nan),
        ("lon", np.nan),
        ("parse_ok", True),
        ("n_detections", 0),
        ("n_masks", 0),
        ("mask_score_mean", 0.0),
        (CANOPY_METRIC_NAME, 0.0),
    ]:
        if base_col not in df.columns:
            df[base_col] = default

    ordered_cols = [
        "box",
        "seq",
        "image_path",
        "timestamp",
        "lat",
        "lon",
        "parse_ok",
        "n_detections",
        "n_masks",
        "mask_score_mean",
        CANOPY_METRIC_NAME,
    ]
    return df[ordered_cols]


def _ensure_agg_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    for base_col, default in [
        ("box", ""),
        ("seq", ""),
        ("lat", np.nan),
        ("lon", np.nan),
        ("n_images_at_point", 0),
        (CANOPY_METRIC_NAME, 0.0),
    ]:
        if base_col not in df.columns:
            df[base_col] = default

    ordered_cols = ["box", "seq", "lat", "lon", "n_images_at_point", CANOPY_METRIC_NAME]
    return df[ordered_cols]


def _overlay_output_path(save_dir: Path, image_path: str) -> Path:
    src = Path(image_path)
    digest = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return save_dir / f"{src.stem}__{digest}.jpg"


def _save_segmentation_overlay(
    save_dir: Path,
    image_path: str,
    image_bgr: np.ndarray,
    mask_measurements: Sequence[MaskMeasurement],
) -> None:
    ensure_dir(save_dir)
    canvas = image_bgr.copy()
    overlay_color = np.array([0, 255, 0], dtype=np.uint8)
    fill_alpha = 0.40

    for measurement in mask_measurements:
        mask = np.asarray(measurement.mask, dtype=bool)
        if mask.size == 0 or not np.any(mask):
            continue
        canvas[mask] = (
            (1.0 - fill_alpha) * canvas[mask].astype(np.float32)
            + fill_alpha * overlay_color.astype(np.float32)
        ).astype(np.uint8)
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(canvas, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

    cv2.imwrite(str(_overlay_output_path(save_dir, image_path)), canvas)


def _measure_image_canopy(
    rec: Dict[str, Any],
    detector: PlantBoxDetector,
    segmenter: MobileSamSegmenter,
    overlay_dir: Path | None,
    logger,
) -> Dict[str, Any]:
    image_bgr = load_image_bgr(rec["image_path"])
    if image_bgr is None:
        logger.warning("Unreadable image; zero-filling canopy measurement: %s", rec["image_path"])
        return {
            **rec,
            "n_detections": 0,
            "n_masks": 0,
            "mask_score_mean": 0.0,
            CANOPY_METRIC_NAME: 0.0,
        }

    detections = detector.predict_image(image_bgr)
    mask_measurements = segmenter.segment_boxes(image_bgr=image_bgr, detections=detections) if detections else []
    canopy_size = float(sum(measurement.pixel_count for measurement in mask_measurements))
    mask_score_mean = (
        float(np.mean([measurement.score for measurement in mask_measurements]))
        if mask_measurements
        else 0.0
    )

    if overlay_dir is not None:
        try:
            if not mask_measurements:
                logger.info(
                    "No valid canopy mask for image; saving original image unchanged: %s",
                    rec["image_path"],
                )
            _save_segmentation_overlay(
                save_dir=overlay_dir,
                image_path=rec["image_path"],
                image_bgr=image_bgr,
                mask_measurements=mask_measurements,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save segmentation overlay for %s (%s)", rec["image_path"], exc)

    return {
        **rec,
        "n_detections": int(len(detections)),
        "n_masks": int(len(mask_measurements)),
        "mask_score_mean": mask_score_mean,
        CANOPY_METRIC_NAME: canopy_size,
    }


def process_sequence(
    seq_info: SequenceInfo,
    out_dir: Path | str,
    detector: PlantBoxDetector,
    segmenter: MobileSamSegmenter,
    image_extensions: Sequence[str],
    agg: str,
    gps_round: int,
    save_overlay_images: bool,
    skip_existing: bool,
    logger,
) -> TrialProcessResult:
    seq_out_dir = ensure_dir(out_dir)
    per_image_csv = seq_out_dir / "canopy_per_image.csv"
    agg_points_csv = seq_out_dir / "canopy_agg_points.csv"
    overlay_dir = seq_out_dir / "segmentation_images"

    parsed_records, n_images_total, n_parse_fail = _parse_sequence_images(
        seq_info=seq_info,
        image_extensions=image_extensions,
        logger=logger,
    )
    overlays_ready = True
    if save_overlay_images:
        overlay_count = len(list(overlay_dir.glob("*.jpg"))) if overlay_dir.exists() else 0
        overlays_ready = overlay_dir.exists() and overlay_count >= len(parsed_records)

    use_cache = bool(
        skip_existing
        and per_image_csv.exists()
        and agg_points_csv.exists()
        and overlays_ready
    )
    if use_cache:
        logger.info(
            "Skipping canopy inference and loading existing CSVs | box=%s seq=%s",
            seq_info.box,
            seq_info.seq,
        )
        per_image_df = pd.read_csv(per_image_csv)
        agg_df = pd.read_csv(agg_points_csv)
    else:
        if save_overlay_images and overlay_dir.exists():
            ensure_dir(overlay_dir)

        rows: List[Dict[str, Any]] = []
        for rec in parsed_records:
            rows.append(
                _measure_image_canopy(
                    rec=rec,
                    detector=detector,
                    segmenter=segmenter,
                    overlay_dir=overlay_dir if save_overlay_images else None,
                    logger=logger,
                )
            )

        per_image_df = pd.DataFrame(rows)
        per_image_df = _ensure_per_image_df_columns(per_image_df)
        per_image_df.to_csv(per_image_csv, index=False)

        agg_input_df = per_image_df[
            ["box", "seq", "image_path", "timestamp", "lat", "lon", "parse_ok", CANOPY_METRIC_NAME]
        ].copy()
        agg_df = aggregate_points(agg_input_df, agg=agg, gps_round=gps_round)
        agg_df = _ensure_agg_df_columns(agg_df)
        agg_df.to_csv(agg_points_csv, index=False)

    per_image_df = _ensure_per_image_df_columns(per_image_df)
    agg_df = _ensure_agg_df_columns(agg_df)
    qgis_export = write_qgis_points_from_agg_df(agg_df=agg_df, output_dir=seq_out_dir)
    distribution_export = write_distribution_layers_from_agg_df(agg_df=agg_df, output_dir=seq_out_dir)

    n_images_parsed_ok = len(parsed_records)
    n_detections_total = int(per_image_df["n_detections"].sum()) if not per_image_df.empty else 0
    n_masks_total = int(per_image_df["n_masks"].sum()) if not per_image_df.empty else 0
    n_gps_points = int(len(agg_df))

    logger.info(
        "Canopy stats | box=%s seq=%s | n_images_total=%d | n_images_parsed_ok=%d | n_parse_fail=%d | detections=%d | masks=%d | gps_points=%d",
        seq_info.box,
        seq_info.seq,
        n_images_total,
        n_images_parsed_ok,
        n_parse_fail,
        n_detections_total,
        n_masks_total,
        n_gps_points,
    )
    logger.info(
        "QGIS GPS export | box=%s seq=%s | points=%d | csv=%s | geojson=%s",
        seq_info.box,
        seq_info.seq,
        qgis_export["point_count"],
        qgis_export["csv_path"],
        qgis_export["geojson_path"],
    )
    logger.info(
        "QGIS distribution export | box=%s seq=%s | targets=%d | manifest=%s",
        seq_info.box,
        seq_info.seq,
        distribution_export["target_count"],
        distribution_export["manifest_path"],
    )

    return TrialProcessResult(
        index_rows=[],
        n_images_total=n_images_total,
        n_images_parsed_ok=n_images_parsed_ok,
        n_parse_fail=n_parse_fail,
        n_gps_points=n_gps_points,
        n_detections_total=n_detections_total,
        n_masks_total=n_masks_total,
    )
