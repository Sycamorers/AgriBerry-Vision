from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from canopy_distribution.mobile_sam import MobileSamSegmenter
from canopy_distribution.pipeline import CANOPY_METRIC_NAME, process_sequence as process_canopy_sequence
from canopy_distribution.plant_detect import PlantBoxDetector
from detection_distribution.io_scan import IMAGE_EXTENSIONS, SequenceInfo
from detection_distribution.qgis_export import (
    write_distribution_layers_from_agg_df,
    write_qgis_points_from_agg_df,
)
from detection_distribution.trial_discovery import TrialFolder, discover_trial_folders
from detection_distribution.utils import ensure_dir, select_weights
from detection_distribution.yolo_infer import get_model_class_names, resolve_infer_device
from run_detection_distribution import _process_sequence as process_detection_sequence


DETECTION_CLASSES = ["FL", "G", "W", "P", "R"]
CANOPY_OUTPUT_NAME = "canopy"
BASE_AGG_COLUMNS = ["box", "seq", "lat", "lon", "n_images_at_point"]


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.lstrip("-").replace("-", "_")
    pos_alias = name.replace("_", "-")
    neg_alias_1 = f"--no_{dest}"
    neg_alias_2 = f"--no-{dest}"
    neg_alias_3 = f"--no-{dest.replace('_', '-')}"

    pos_opts = list(dict.fromkeys([name, pos_alias]))
    neg_opts = list(dict.fromkeys([neg_alias_1, neg_alias_2, neg_alias_3]))

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(*pos_opts, dest=dest, action="store_true", help=help_text)
    group.add_argument(*neg_opts, dest=dest, action="store_false", help=f"Disable {dest}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AgriBerry Vision detection and canopy processing over data/ and "
            "export box-level QGIS-ready layers into QGIS_out/{date}/{box}."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Input root directory containing date/box/trial folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "QGIS_out",
        help="Final output root directory for box-level QGIS layers",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Fruit detection weights path. If omitted, auto-select from ./ckpt/",
    )
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=PROJECT_ROOT / "ckpt" / "plant.pt",
        help="YOLO plant detector weights path for canopy processing",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "ckpt" / "mobile_sam.pt",
        help="MobileSAM checkpoint path",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=sorted(IMAGE_EXTENSIONS),
        help="Image extensions to treat as valid input files",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0" or "cpu"')
    parser.add_argument("--batch", type=int, default=16, help="YOLO batch size for fruit detection")
    parser.add_argument("--agg", type=str, choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--gps_round", type=int, default=6, help="Decimals for GPS point rounding")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output root before writing")
    _add_bool_arg(parser, "--keep_temp", default=False, help_text="Keep temporary trial-level outputs")
    return parser.parse_args()


def configure_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)

    logger = logging.getLogger("qgis_out_export")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def normalize_extensions(extensions: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ext in extensions:
        ext_text = str(ext).strip().lower()
        if not ext_text:
            continue
        if not ext_text.startswith("."):
            ext_text = f".{ext_text}"
        if ext_text not in seen:
            normalized.append(ext_text)
            seen.add(ext_text)
    return normalized


def resolve_fruit_weights(weights: str | None) -> Path:
    if weights:
        return select_weights(weights, ckpt_dir=PROJECT_ROOT / "ckpt")

    fruit_weights = PROJECT_ROOT / "ckpt" / "fruit.pt"
    if fruit_weights.exists():
        return fruit_weights

    return select_weights(None, ckpt_dir=PROJECT_ROOT / "ckpt")


def _load_trial_agg_csvs(
    root: Path,
    trial: TrialFolder,
    filename: str,
) -> list[pd.DataFrame]:
    csv_path = root / trial.output_rel_path / filename
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    return [df]


def _merge_agg_frames(
    frames: Sequence[pd.DataFrame],
    *,
    box_name: str,
    value_columns: Sequence[str],
    agg: str,
) -> pd.DataFrame:
    out_columns = [*BASE_AGG_COLUMNS, *value_columns]
    if not frames:
        return pd.DataFrame(columns=out_columns)

    prepared: list[pd.DataFrame] = []
    for frame in frames:
        work = frame.copy()
        for column in BASE_AGG_COLUMNS:
            if column not in work.columns:
                work[column] = 0 if column == "n_images_at_point" else ""
        for column in value_columns:
            if column not in work.columns:
                work[column] = 0.0
        work = work[[*BASE_AGG_COLUMNS, *value_columns]]
        work["box"] = box_name
        work["seq"] = ""
        work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
        work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
        work["n_images_at_point"] = pd.to_numeric(work["n_images_at_point"], errors="coerce").fillna(0).astype(int)
        for column in value_columns:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        work = work.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        if not work.empty:
            prepared.append(work)

    if not prepared:
        return pd.DataFrame(columns=out_columns)

    merged = pd.concat(prepared, ignore_index=True)
    group_cols = ["lat", "lon"]
    rows: list[dict[str, object]] = []

    for (lat, lon), group in merged.groupby(group_cols, sort=True):
        weights = group["n_images_at_point"].astype(float)
        row: dict[str, object] = {
            "box": box_name,
            "seq": "",
            "lat": float(lat),
            "lon": float(lon),
            "n_images_at_point": int(group["n_images_at_point"].sum()),
        }

        for column in value_columns:
            values = group[column].astype(float)
            if agg == "mean":
                denom = float(weights.sum())
                row[column] = float((values * weights).sum() / denom) if denom > 0 else 0.0
            elif agg == "sum":
                row[column] = int(round(float(values.sum())))
            else:
                row[column] = int(round(float(values.max()))) if not values.empty else 0

        rows.append(row)

    return pd.DataFrame(rows, columns=out_columns).sort_values(["lat", "lon"]).reset_index(drop=True)


def _combine_box_level_metrics(
    detection_df: pd.DataFrame,
    canopy_df: pd.DataFrame,
    *,
    box_name: str,
) -> pd.DataFrame:
    det = detection_df.copy()
    can = canopy_df.copy()

    if det.empty:
        det = pd.DataFrame(columns=[*BASE_AGG_COLUMNS, *DETECTION_CLASSES])
    if can.empty:
        can = pd.DataFrame(columns=[*BASE_AGG_COLUMNS, CANOPY_OUTPUT_NAME])

    if CANOPY_METRIC_NAME in can.columns and CANOPY_OUTPUT_NAME not in can.columns:
        can = can.rename(columns={CANOPY_METRIC_NAME: CANOPY_OUTPUT_NAME})

    det = det[[*BASE_AGG_COLUMNS, *DETECTION_CLASSES]] if not det.empty else det
    can = can[[*BASE_AGG_COLUMNS, CANOPY_OUTPUT_NAME]] if not can.empty else can

    det = det.rename(columns={"n_images_at_point": "n_images_at_point_det"})
    can = can.rename(columns={"n_images_at_point": "n_images_at_point_can"})

    merged = det.merge(
        can,
        on=["box", "seq", "lat", "lon"],
        how="outer",
    )
    if merged.empty:
        return pd.DataFrame(columns=[*BASE_AGG_COLUMNS, *DETECTION_CLASSES, CANOPY_OUTPUT_NAME])

    merged["box"] = box_name
    merged["seq"] = ""
    merged["n_images_at_point_det"] = pd.to_numeric(merged["n_images_at_point_det"], errors="coerce").fillna(0)
    merged["n_images_at_point_can"] = pd.to_numeric(merged["n_images_at_point_can"], errors="coerce").fillna(0)
    merged["n_images_at_point"] = (
        merged[["n_images_at_point_det", "n_images_at_point_can"]].max(axis=1).round().astype(int)
    )

    for column in [*DETECTION_CLASSES, CANOPY_OUTPUT_NAME]:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    out_columns = [*BASE_AGG_COLUMNS, *DETECTION_CLASSES, CANOPY_OUTPUT_NAME]
    return merged[out_columns].sort_values(["lat", "lon"]).reset_index(drop=True)


def _write_box_outputs(
    *,
    box_df: pd.DataFrame,
    box_output_dir: Path,
) -> dict[str, object]:
    ensure_dir(box_output_dir)
    qgis_points = write_qgis_points_from_agg_df(agg_df=box_df, output_dir=box_output_dir)
    distribution = write_distribution_layers_from_agg_df(
        agg_df=box_df,
        output_dir=box_output_dir,
        dirname="",
    )

    agg_csv_path = box_output_dir / "box_agg_points.csv"
    box_df.to_csv(agg_csv_path, index=False)

    return {
        "gps_csv_path": qgis_points["csv_path"],
        "gps_geojson_path": qgis_points["geojson_path"],
        "manifest_path": distribution["manifest_path"],
        "point_count": qgis_points["point_count"],
        "target_count": distribution["target_count"],
        "agg_csv_path": str(agg_csv_path.resolve()),
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Re-run with --overwrite.")
        shutil.rmtree(output_root)

    ensure_dir(output_root)
    logger = configure_logger(output_root / "qgis_out.log")
    image_extensions = normalize_extensions(args.ext)
    if not image_extensions:
        raise ValueError("At least one valid image extension must be provided via --ext")

    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    logger.info("Image extensions: %s", image_extensions)

    trials = discover_trial_folders(input_root)
    if not trials:
        raise RuntimeError(f"No trial folders found under {input_root}")
    logger.info("Discovered %d trial folder(s).", len(trials))

    temp_root = Path(tempfile.mkdtemp(prefix="qgis_out_tmp_", dir=str(PROJECT_ROOT)))
    detection_tmp_root = ensure_dir(temp_root / "detection")
    canopy_tmp_root = ensure_dir(temp_root / "canopy")
    logger.info("Temporary root: %s", temp_root)

    try:
        resolved_device = resolve_infer_device(args.device, logger=logger)

        fruit_weights = resolve_fruit_weights(args.weights)
        fruit_model = YOLO(str(fruit_weights))
        fruit_class_names = get_model_class_names(fruit_model.names)
        missing_classes = [name for name in DETECTION_CLASSES if name not in fruit_class_names]
        if missing_classes:
            raise RuntimeError(
                f"Fruit model is missing expected classes: {missing_classes}. "
                f"Found classes: {fruit_class_names}"
            )

        canopy_detector = PlantBoxDetector(
            weights=args.detector_weights,
            device=resolved_device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            logger=logger,
        )
        canopy_segmenter = MobileSamSegmenter(
            checkpoint=args.sam_checkpoint,
            device=resolved_device,
            logger=logger,
        )

        detection_args = argparse.Namespace(
            out_dir=None,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=resolved_device,
            batch=args.batch,
            agg=args.agg,
            gps_round=args.gps_round,
            save_pred_images=False,
            skip_existing=False,
        )

        for index, trial in enumerate(trials, start=1):
            logger.info(
                "Processing trial %d/%d | date=%s box=%s trial=%s",
                index,
                len(trials),
                trial.date,
                trial.box,
                trial.trial,
            )
            seq_info = SequenceInfo(box=trial.box, seq=trial.trial, seq_dir=trial.trial_dir)

            detection_args.out_dir = ensure_dir(detection_tmp_root / trial.date)
            process_detection_sequence(
                seq_info=seq_info,
                args=detection_args,
                model=fruit_model,
                class_names=fruit_class_names,
                logger=logger,
            )

            process_canopy_sequence(
                seq_info=seq_info,
                out_dir=ensure_dir(canopy_tmp_root / trial.output_rel_path),
                detector=canopy_detector,
                segmenter=canopy_segmenter,
                image_extensions=image_extensions,
                agg=args.agg,
                gps_round=args.gps_round,
                save_overlay_images=False,
                skip_existing=False,
                logger=logger,
            )

        summary_rows: list[dict[str, object]] = []
        grouped_trials: dict[tuple[str, str], list[TrialFolder]] = {}
        for trial in trials:
            grouped_trials.setdefault((trial.date, trial.box), []).append(trial)

        for (date, box), box_trials in sorted(grouped_trials.items()):
            logger.info("Aggregating box-level QGIS layers | date=%s box=%s | trials=%d", date, box, len(box_trials))
            detection_frames: list[pd.DataFrame] = []
            canopy_frames: list[pd.DataFrame] = []
            for trial in box_trials:
                detection_frames.extend(
                    _load_trial_agg_csvs(detection_tmp_root, trial, "detections_agg_points.csv")
                )
                canopy_frames.extend(
                    _load_trial_agg_csvs(canopy_tmp_root, trial, "canopy_agg_points.csv")
                )

            detection_box_df = _merge_agg_frames(
                detection_frames,
                box_name=box,
                value_columns=DETECTION_CLASSES,
                agg=args.agg,
            )
            canopy_box_df = _merge_agg_frames(
                canopy_frames,
                box_name=box,
                value_columns=[CANOPY_METRIC_NAME],
                agg=args.agg,
            )
            box_df = _combine_box_level_metrics(
                detection_df=detection_box_df,
                canopy_df=canopy_box_df,
                box_name=box,
            )

            box_output_dir = ensure_dir(output_root / date / box)
            export_summary = _write_box_outputs(
                box_df=box_df,
                box_output_dir=box_output_dir,
            )
            summary_rows.append(
                {
                    "date": date,
                    "box": box,
                    "trial_count": len(box_trials),
                    "point_count": export_summary["point_count"],
                    "target_count": export_summary["target_count"],
                    "gps_csv_path": export_summary["gps_csv_path"],
                    "gps_geojson_path": export_summary["gps_geojson_path"],
                    "manifest_path": export_summary["manifest_path"],
                    "agg_csv_path": export_summary["agg_csv_path"],
                }
            )

        summary_df = pd.DataFrame(
            summary_rows,
            columns=[
                "date",
                "box",
                "trial_count",
                "point_count",
                "target_count",
                "gps_csv_path",
                "gps_geojson_path",
                "manifest_path",
                "agg_csv_path",
            ],
        )
        summary_csv = output_root / "index.csv"
        summary_df.to_csv(summary_csv, index=False)

        run_summary = {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "temporary_root": str(temp_root),
            "fruit_weights": str(fruit_weights),
            "fruit_classes": fruit_class_names,
            "plant_detector_weights": str(args.detector_weights.expanduser().resolve()),
            "sam_checkpoint": str(args.sam_checkpoint.expanduser().resolve()),
            "agg": args.agg,
            "gps_round": int(args.gps_round),
            "boxes_exported": int(len(summary_rows)),
            "index_csv": str(summary_csv.resolve()),
        }
        summary_json = output_root / "run_summary.json"
        summary_json.write_text(json.dumps(run_summary, ensure_ascii=True, indent=2), encoding="utf-8")

        logger.info("Completed QGIS export. Box-level outputs written: %d", len(summary_rows))
        logger.info("Index CSV: %s", summary_csv)
        logger.info("Run summary JSON: %s", summary_json)
    finally:
        if args.keep_temp:
            logger.info("Keeping temporary root: %s", temp_root)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
