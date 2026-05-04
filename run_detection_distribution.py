from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detection_distribution.aggregate import aggregate_points
from detection_distribution.gps_parse import parse_image_path
from detection_distribution.io_scan import SequenceInfo, collect_image_files
from detection_distribution.qgis_export import (
    write_distribution_layers_from_agg_df,
    write_qgis_points_from_agg_df,
)
from detection_distribution.trial_discovery import discover_trial_folders
from detection_distribution.utils import ensure_dir, select_weights, setup_logger
from detection_distribution.yolo_infer import (
    get_model_class_names,
    resolve_infer_device,
    run_yolo_inference,
)


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
            "Run YOLO on GPS-tagged image sequences, aggregate repeated GPS samples, "
            "and export QGIS-ready detection layers."
        )
    )

    parser.add_argument("--data_dir", type=str, default="data", help="Input data directory")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/detection_distribution",
        help="Output root directory",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="YOLO weights path. If omitted, auto-select from ./ckpt/",
    )

    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0" or "cpu"')
    parser.add_argument("--batch", type=int, default=16, help="YOLO batch size")

    parser.add_argument("--agg", type=str, choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--gps_round", type=int, default=6, help="Decimals for GPS point rounding")

    _add_bool_arg(
        parser,
        "--save_pred_images",
        default=True,
        help_text="Save per-image inference visualizations with bounding boxes",
    )
    _add_bool_arg(
        parser,
        "--skip_existing",
        default=True,
        help_text="Reuse existing CSVs if they already exist",
    )

    return parser.parse_args()


def _parse_sequence_images(
    seq_info: SequenceInfo,
    logger,
) -> Tuple[List[Dict[str, Any]], int, int]:
    image_paths = collect_image_files(seq_info.seq_dir)
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
            }
        )

    return parsed_records, len(image_paths), parse_fail_count


def _ensure_detection_df_columns(df: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    for base_col, default in [
        ("box", ""),
        ("seq", ""),
        ("image_path", ""),
        ("timestamp", ""),
        ("lat", np.nan),
        ("lon", np.nan),
        ("parse_ok", True),
    ]:
        if base_col not in df.columns:
            df[base_col] = default

    for class_name in class_names:
        if class_name not in df.columns:
            df[class_name] = 0

    ordered_cols = ["box", "seq", "image_path", "timestamp", "lat", "lon", "parse_ok", *class_names]
    return df[ordered_cols]


def _ensure_agg_df_columns(df: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    for base_col, default in [
        ("box", ""),
        ("seq", ""),
        ("lat", np.nan),
        ("lon", np.nan),
        ("n_images_at_point", 0),
    ]:
        if base_col not in df.columns:
            df[base_col] = default

    for class_name in class_names:
        if class_name not in df.columns:
            df[class_name] = 0.0

    ordered_cols = ["box", "seq", "lat", "lon", "n_images_at_point", *class_names]
    return df[ordered_cols]


def _log_sequence_sanity(
    box: str,
    seq: str,
    n_images_total: int,
    n_images_parsed_ok: int,
    n_parse_fail: int,
    parsed_records: List[Dict[str, Any]],
    agg_df: pd.DataFrame,
    class_names: List[str],
    logger,
) -> None:
    logger.info(
        "Sequence stats | box=%s seq=%s | n_images_total=%d | n_images_parsed_ok=%d | n_parse_fail=%d",
        box,
        seq,
        n_images_total,
        n_images_parsed_ok,
        n_parse_fail,
    )

    if parsed_records:
        lats = np.array([rec["lat"] for rec in parsed_records], dtype=float)
        lons = np.array([rec["lon"] for rec in parsed_records], dtype=float)
        logger.info(
            "GPS bounds | box=%s seq=%s | lat[min,max]=[%.8f, %.8f] lon[min,max]=[%.8f, %.8f]",
            box,
            seq,
            float(np.min(lats)),
            float(np.max(lats)),
            float(np.min(lons)),
            float(np.max(lons)),
        )
    else:
        logger.info("GPS bounds | box=%s seq=%s | no parsed points", box, seq)

    for class_name in class_names:
        values = agg_df[class_name].to_numpy(dtype=float) if class_name in agg_df.columns else np.array([], dtype=float)
        n_nonzero = int(np.sum(values > 0)) if values.size > 0 else 0
        max_val = float(np.max(values)) if values.size > 0 else 0.0
        mean_val = float(np.mean(values)) if values.size > 0 else 0.0
        logger.info(
            "Class stats | box=%s seq=%s class=%s | nonzero_points=%d max=%.5f mean=%.5f",
            box,
            seq,
            class_name,
            n_nonzero,
            max_val,
            mean_val,
        )


def _process_sequence(
    seq_info: SequenceInfo,
    args: argparse.Namespace,
    model: YOLO,
    class_names: List[str],
    logger,
) -> List[Dict[str, Any]]:
    box = seq_info.box
    seq = seq_info.seq

    seq_out_dir = ensure_dir(Path(args.out_dir) / box / seq)
    per_image_csv = seq_out_dir / "detections_per_image.csv"
    agg_points_csv = seq_out_dir / "detections_agg_points.csv"
    pred_vis_dir = seq_out_dir / "inference_images"

    parsed_records, n_images_total, n_parse_fail = _parse_sequence_images(seq_info, logger)

    if args.save_pred_images and pred_vis_dir.exists():
        pred_image_count = len(list(pred_vis_dir.glob("*.jpg")))
    else:
        pred_image_count = 0
    pred_images_ready = (not args.save_pred_images) or (
        pred_vis_dir.exists() and pred_image_count >= len(parsed_records)
    )
    use_cache = bool(
        args.skip_existing
        and per_image_csv.exists()
        and agg_points_csv.exists()
        and pred_images_ready
    )
    if use_cache:
        logger.info(
            "Skipping inference and loading existing CSVs | box=%s seq=%s",
            box,
            seq,
        )
        detections_df = pd.read_csv(per_image_csv)
        agg_df = pd.read_csv(agg_points_csv)
    else:
        if args.skip_existing and per_image_csv.exists() and agg_points_csv.exists() and args.save_pred_images and not pred_images_ready:
            logger.info(
                "Existing CSVs found but prediction images are incomplete (%d/%d). Re-running inference to export bbox visualizations.",
                pred_image_count,
                len(parsed_records),
            )
        if parsed_records:
            detections_df = run_yolo_inference(
                model=model,
                image_records=parsed_records,
                class_names=class_names,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                batch=args.batch,
                save_pred_dir=pred_vis_dir if args.save_pred_images else None,
                logger=logger,
            )
        else:
            detections_df = pd.DataFrame(columns=["box", "seq", "image_path", "timestamp", "lat", "lon", "parse_ok", *class_names])

        detections_df = _ensure_detection_df_columns(detections_df, class_names)
        detections_df.to_csv(per_image_csv, index=False)

        agg_df = aggregate_points(detections_df, agg=args.agg, gps_round=args.gps_round)
        agg_df = _ensure_agg_df_columns(agg_df, class_names)
        agg_df.to_csv(agg_points_csv, index=False)

    detections_df = _ensure_detection_df_columns(detections_df, class_names)
    agg_df = _ensure_agg_df_columns(agg_df, class_names)
    qgis_export = write_qgis_points_from_agg_df(agg_df=agg_df, output_dir=seq_out_dir)
    distribution_export = write_distribution_layers_from_agg_df(agg_df=agg_df, output_dir=seq_out_dir)

    n_images_parsed_ok = len(parsed_records)
    _log_sequence_sanity(
        box=box,
        seq=seq,
        n_images_total=n_images_total,
        n_images_parsed_ok=n_images_parsed_ok,
        n_parse_fail=n_parse_fail,
        parsed_records=parsed_records,
        agg_df=agg_df,
        class_names=class_names,
        logger=logger,
    )
    logger.info(
        "QGIS GPS export | box=%s seq=%s | points=%d | csv=%s | geojson=%s",
        box,
        seq,
        qgis_export["point_count"],
        qgis_export["csv_path"],
        qgis_export["geojson_path"],
    )
    logger.info(
        "QGIS distribution export | box=%s seq=%s | targets=%d | manifest=%s",
        box,
        seq,
        distribution_export["target_count"],
        distribution_export["manifest_path"],
    )

    n_images_for_index = int(len(detections_df))
    n_points_for_index = int(len(agg_df))

    index_rows: List[Dict[str, Any]] = []
    for target in distribution_export["targets"]:
        index_rows.append(
            {
                "box": box,
                "seq": seq,
                "class_name": target["target"],
                "csv_path": target["csv_path"],
                "geojson_path": target["geojson_path"],
                "point_count": target["point_count"],
                "n_images": n_images_for_index,
                "n_gps_points": n_points_for_index,
                "agg": args.agg,
                "conf": args.conf,
                "imgsz": args.imgsz,
            }
        )

    return index_rows


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    out_dir = ensure_dir(args.out_dir)
    args.device = resolve_infer_device(args.device, logger=logger)

    weights_path = select_weights(args.weights, ckpt_dir=PROJECT_ROOT / "ckpt")
    logger.info("Using weights: %s", weights_path)

    model = YOLO(str(weights_path))
    class_names = get_model_class_names(model.names)

    logger.info("Loaded YOLO model with %d classes.", len(class_names))

    trials = discover_trial_folders(Path(args.data_dir))
    logger.info("Discovered %d trial folder(s).", len(trials))

    index_rows: List[Dict[str, Any]] = []
    for trial in trials:
        logger.info(
            "Processing trial | date=%s box=%s trial=%s",
            trial.date,
            trial.box,
            trial.trial,
        )
        seq_info = SequenceInfo(box=trial.box, seq=trial.trial, seq_dir=trial.trial_dir)
        trial_args = argparse.Namespace(**vars(args))
        trial_args.out_dir = out_dir / trial.date
        try:
            rows = (
                _process_sequence(
                    seq_info=seq_info,
                    args=trial_args,
                    model=model,
                    class_names=class_names,
                    logger=logger,
                )
            )
            for row in rows:
                row["date"] = trial.date
            index_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Trial failed but continuing | date=%s box=%s trial=%s | error=%s",
                trial.date,
                trial.box,
                trial.trial,
                exc,
            )

    index_df = pd.DataFrame(
        index_rows,
        columns=[
            "date",
            "box",
            "seq",
            "class_name",
            "csv_path",
            "geojson_path",
            "point_count",
            "n_images",
            "n_gps_points",
            "agg",
            "conf",
            "imgsz",
        ],
    )
    index_csv = out_dir / "index.csv"
    index_df.to_csv(index_csv, index=False)

    logger.info("Completed. Wrote index: %s", index_csv)


if __name__ == "__main__":
    main()
