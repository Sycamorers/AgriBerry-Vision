from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_model_class_names(model_names: Any) -> List[str]:
    """Normalize ultralytics model.names into an index-aligned class list."""
    if isinstance(model_names, dict):
        if not model_names:
            return []
        int_keys = [int(k) for k in model_names.keys()]
        max_idx = max(int_keys)
        return [str(model_names.get(i, f"class_{i}")) for i in range(max_idx + 1)]

    if isinstance(model_names, (list, tuple)):
        return [str(v) for v in model_names]

    raise TypeError(f"Unsupported model.names type: {type(model_names)!r}")


def resolve_infer_device(
    device: str | int = "0",
    logger: logging.Logger | None = None,
) -> str:
    """Resolve device string for ultralytics and log runtime CUDA details."""
    requested = str(device).strip() if device is not None else "0"
    requested_lower = requested.lower()

    try:
        import torch
    except Exception:  # noqa: BLE001
        if requested_lower != "cpu" and logger:
            logger.warning("PyTorch unavailable for device probing; falling back to CPU.")
        return "cpu"

    if requested_lower == "cpu":
        return "cpu"

    # Accept explicit CUDA strings, e.g. cuda:0.
    if requested_lower.startswith("cuda"):
        if not torch.cuda.is_available():
            if logger:
                logger.warning("CUDA requested (%s) but torch.cuda.is_available() is False. Using CPU.", requested)
            return "cpu"
        resolved = requested_lower
    # Numeric GPU index, e.g. "0".
    elif requested.isdigit():
        if not torch.cuda.is_available():
            if logger:
                logger.warning("GPU index %s requested but CUDA is unavailable. Using CPU.", requested)
            return "cpu"
        gpu_idx = int(requested)
        n_gpu = int(torch.cuda.device_count())
        if n_gpu <= 0:
            if logger:
                logger.warning("No CUDA devices found. Using CPU.")
            return "cpu"
        if gpu_idx >= n_gpu:
            if logger:
                logger.warning(
                    "Requested GPU index %d out of range (n_gpu=%d). Using cuda:0.",
                    gpu_idx,
                    n_gpu,
                )
            gpu_idx = 0
        resolved = f"cuda:{gpu_idx}"
    else:
        # Pass through values like mps when available; otherwise let ultralytics handle or fallback.
        resolved = requested_lower

    if resolved.startswith("cuda"):
        try:
            idx = int(resolved.split(":")[1]) if ":" in resolved else 0
            name = torch.cuda.get_device_name(idx)
            if logger:
                logger.info("Resolved inference device: %s (%s)", resolved, name)
        except Exception:  # noqa: BLE001
            if logger:
                logger.info("Resolved inference device: %s", resolved)
    else:
        if logger:
            logger.info("Resolved inference device: %s", resolved)

    return resolved


def _count_classes_from_result(result: Any, n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int32)

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return counts

    cls_tensor = getattr(boxes, "cls", None)
    if cls_tensor is None:
        return counts

    try:
        cls_ids = cls_tensor.detach().cpu().numpy().astype(int)
    except Exception:  # noqa: BLE001
        return counts

    if cls_ids.size == 0:
        return counts

    valid_ids = cls_ids[(cls_ids >= 0) & (cls_ids < n_classes)]
    if valid_ids.size == 0:
        return counts

    bincount = np.bincount(valid_ids, minlength=n_classes)
    counts[: len(bincount)] = bincount[:n_classes]
    return counts


def _canonicalize_image_for_infer(img: Any) -> np.ndarray | None:
    """Return a strict uint8 HxWx3 contiguous ndarray suitable for cv2/ultralytics."""
    if img is None:
        return None

    # Handle cv2.UMat-like objects.
    if hasattr(img, "get") and callable(getattr(img, "get")):
        try:
            img = img.get()
        except Exception:  # noqa: BLE001
            return None

    try:
        arr = np.asarray(img)
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return None
    if arr.dtype == np.object_:
        return None

    if arr.ndim == 2:
        try:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception:  # noqa: BLE001
            return None
    elif arr.ndim == 3 and arr.shape[2] == 4:
        try:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        except Exception:  # noqa: BLE001
            return None
    elif not (arr.ndim == 3 and arr.shape[2] == 3):
        return None

    if arr.dtype != np.uint8:
        try:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
            arr = arr.astype(np.uint8)
        except Exception:  # noqa: BLE001
            return None

    # Force plain owned C-order storage.
    try:
        arr = np.array(arr, dtype=np.uint8, copy=True, order="C")
    except Exception:  # noqa: BLE001
        return None

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    return arr


def run_yolo_inference(
    model: Any,
    image_records: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | int = "0",
    batch: int = 16,
    save_pred_dir: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Run YOLO prediction and return per-image class count DataFrame.

    This implementation uses batch inference over preloaded numpy BGR images.
    It falls back to per-image mode only for failed batches.
    """
    if batch <= 0:
        raise ValueError("batch must be > 0")
    try:
        imgsz_int = int(imgsz)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"imgsz must be an integer-like value, got {imgsz!r}") from exc
    if imgsz_int <= 0:
        raise ValueError(f"imgsz must be > 0, got {imgsz_int}")
    conf_float = float(conf)
    iou_float = float(iou)
    if not math.isfinite(conf_float) or not math.isfinite(iou_float):
        raise ValueError(f"conf/iou must be finite numbers, got conf={conf!r}, iou={iou!r}")
    batch_size = int(batch)

    # Keep OpenCV image objects as regular numpy arrays for predictable inference behavior.
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:  # noqa: BLE001
        pass

    resolved_device = resolve_infer_device(device=device, logger=logger)
    n_classes = len(class_names)
    rows: List[Dict[str, Any]] = []
    pred_dir = Path(save_pred_dir) if save_pred_dir is not None else None
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Prefer setting model device once to avoid implicit per-call device moves.
    try:
        if hasattr(model, "to"):
            model.to(resolved_device)
    except Exception:  # noqa: BLE001
        pass

    unreadable_paths: set[str] = set()
    unreadable_count = 0
    pred_saved_count = 0
    pred_failed_count = 0
    batch_fallback_count = 0
    warning_printed = 0
    warning_cap = 20
    first_predict_fail_logged = False

    def _short_reason(reason: str) -> str:
        text = str(reason).strip()
        if not text:
            return "unknown error"
        return text.splitlines()[0]

    def _warn_unreadable(path: str, reason: str) -> None:
        nonlocal warning_printed
        if not logger:
            return
        if warning_printed < warning_cap:
            logger.warning(
                "Skipping unreadable image with zero counts: %s | reason=%s",
                path,
                _short_reason(reason),
            )
            warning_printed += 1
        elif warning_printed == warning_cap:
            logger.warning("Additional unreadable-image warnings suppressed...")
            warning_printed += 1

    def _load_image(path: str) -> np.ndarray | None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return _canonicalize_image_for_infer(img)

    def _make_row(rec: Dict[str, Any], counts: np.ndarray) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "box": rec["box"],
            "seq": rec["seq"],
            "image_path": rec["image_path"],
            "timestamp": rec["timestamp"],
            "lat": float(rec["lat"]),
            "lon": float(rec["lon"]),
            "parse_ok": True,
        }
        for cls_idx, cls_name in enumerate(class_names):
            row[cls_name] = int(counts[cls_idx])
        return row

    def _pred_image_path(image_path: str) -> Path:
        src = Path(image_path)
        digest = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
        return (pred_dir / f"{src.stem}__{digest}.jpg") if pred_dir is not None else Path()

    def _save_pred_image(image_path: str, rendered_bgr: np.ndarray) -> None:
        nonlocal pred_saved_count, pred_failed_count
        if pred_dir is None:
            return
        try:
            if rendered_bgr is None:
                pred_failed_count += 1
                return
            arr = np.asarray(rendered_bgr)
            if arr.size == 0 or arr.dtype == np.object_:
                pred_failed_count += 1
                return
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif not (arr.ndim == 3 and arr.shape[2] == 3):
                pred_failed_count += 1
                return

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            arr = np.ascontiguousarray(arr)
            out_path = _pred_image_path(image_path)
            ok = cv2.imwrite(str(out_path), arr)
            if ok:
                pred_saved_count += 1
            else:
                pred_failed_count += 1
        except Exception:  # noqa: BLE001
            pred_failed_count += 1

    def _save_failed_overlay(image_path: str, image_np: np.ndarray) -> None:
        # Keep this path strictly non-fatal: just persist the raw frame if possible.
        if pred_dir is not None:
            _save_pred_image(image_path, image_np)

    def _predict_single_with_retries(image_np: np.ndarray, image_path: str) -> tuple[Any | None, str]:
        canonical = _canonicalize_image_for_infer(image_np)
        if canonical is None:
            return None, "failed to canonicalize image before single-image predict"

        path_source = str(image_path) if image_path else ""
        attempts: List[Tuple[str, Dict[str, Any], bool]] = [
            ("np", {"source": canonical}, False),
            ("np_list", {"source": [canonical]}, False),
        ]
        if path_source:
            attempts.append(("path", {"source": path_source}, False))
        attempts.append(("np_reset", {"source": canonical}, True))
        if path_source:
            attempts.append(("path_reset", {"source": path_source}, True))
        errors: List[str] = []

        for name, kwargs, reset_predictor in attempts:
            if reset_predictor:
                try:
                    if hasattr(model, "predictor"):
                        model.predictor = None
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}:predictor_reset:{_short_reason(exc)}")
                    continue
            try:
                result_list = model.predict(
                    conf=conf_float,
                    iou=iou_float,
                    imgsz=imgsz_int,
                    device=resolved_device,
                    verbose=False,
                    batch=1,
                    stream=False,
                    **kwargs,
                )
                result0 = result_list[0] if len(result_list) > 0 else None
                return result0, ""
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}:{_short_reason(exc)}")

        return None, " | ".join(errors) if errors else "all inference retries failed"

    def _predict_batch(images: List[np.ndarray], image_paths: List[str]) -> tuple[List[Any] | None, str]:
        if not images:
            return [], ""
        canonical_images: List[np.ndarray] = []
        for img in images:
            canonical = _canonicalize_image_for_infer(img)
            if canonical is None:
                return None, "failed to canonicalize at least one batch image"
            canonical_images.append(canonical)

        batch_predict_size = min(max(1, batch_size), len(images))
        attempts: List[Tuple[str, Dict[str, Any], bool]] = [
            ("batch_np", {"source": canonical_images}, False),
        ]
        if image_paths:
            attempts.append(("batch_path", {"source": image_paths}, False))
        attempts.append(("batch_np_reset", {"source": canonical_images}, True))
        if image_paths:
            attempts.append(("batch_path_reset", {"source": image_paths}, True))

        errors: List[str] = []
        for name, kwargs, reset_predictor in attempts:
            if reset_predictor:
                try:
                    if hasattr(model, "predictor"):
                        model.predictor = None
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}:predictor_reset:{_short_reason(exc)}")
                    continue
            try:
                result_list = model.predict(
                    conf=conf_float,
                    iou=iou_float,
                    imgsz=imgsz_int,
                    device=resolved_device,
                    verbose=False,
                    batch=batch_predict_size,
                    stream=False,
                    **kwargs,
                )
                out = list(result_list)
                if len(out) != len(images):
                    errors.append(f"{name}:result length mismatch: got {len(out)} expected {len(images)}")
                    continue
                return out, ""
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}:{_short_reason(exc)}")

        return None, " | ".join(errors) if errors else "all batch retries failed"

    if logger:
        logger.info(
            "Starting YOLO inference | n_images=%d | conf=%.3f | iou=%.3f | imgsz=%d | device=%s | batch=%d | mode=batch-with-fallback",
            len(image_records),
            conf_float,
            iou_float,
            imgsz_int,
            str(resolved_device),
            batch_size,
        )

    image_records = list(image_records)
    total_batches = (len(image_records) + batch_size - 1) // batch_size if image_records else 0
    for start in tqdm(
        range(0, len(image_records), batch_size),
        total=total_batches,
        desc="YOLO inference",
        unit="batch",
        leave=False,
    ):
        batch_recs = image_records[start : start + batch_size]
        batch_rows: List[Dict[str, Any] | None] = [None] * len(batch_recs)
        loaded_entries: List[Tuple[int, Dict[str, Any], np.ndarray]] = []

        for local_idx, rec in enumerate(batch_recs):
            path = rec["image_path"]
            if path in unreadable_paths:
                unreadable_count += 1
                batch_rows[local_idx] = _make_row(rec, np.zeros(n_classes, dtype=np.int32))
                continue

            image_np = _load_image(path)
            if image_np is None:
                unreadable_paths.add(path)
                unreadable_count += 1
                _warn_unreadable(path, "cv2.imread returned empty")
                batch_rows[local_idx] = _make_row(rec, np.zeros(n_classes, dtype=np.int32))
                continue

            loaded_entries.append((local_idx, rec, image_np))

        batch_results: List[Any] | None = None
        batch_error = ""
        if loaded_entries:
            batch_images = [entry[2] for entry in loaded_entries]
            batch_paths = [str(entry[1]["image_path"]) for entry in loaded_entries]
            batch_results, batch_error = _predict_batch(batch_images, image_paths=batch_paths)
            if batch_results is None:
                batch_fallback_count += 1
                if logger:
                    logger.warning(
                        "Batch inference failed. Falling back to per-image mode for this batch. error=%s",
                        batch_error,
                    )

        for idx_in_loaded, (local_idx, rec, image_np) in enumerate(loaded_entries):
            path = rec["image_path"]
            result0 = None
            failure_reason = ""

            if batch_results is not None:
                result0 = batch_results[idx_in_loaded]
            else:
                result0, single_error = _predict_single_with_retries(image_np=image_np, image_path=str(path))
                failure_reason = f"batch:{batch_error} | single:{single_error}" if batch_error else single_error

            if result0 is None:
                unreadable_paths.add(path)
                unreadable_count += 1
                if logger and not first_predict_fail_logged:
                    first_predict_fail_logged = True
                    logger.warning(
                        "First predict failure diagnostics | path=%s | type=%s | shape=%s | dtype=%s | c_contig=%s",
                        path,
                        type(image_np).__name__,
                        tuple(image_np.shape) if hasattr(image_np, "shape") else "n/a",
                        str(getattr(image_np, "dtype", "n/a")),
                        bool(getattr(image_np, "flags", {}).c_contiguous) if hasattr(image_np, "flags") else False,
                    )
                _warn_unreadable(path, failure_reason or "all inference retries failed")
                if pred_dir is not None:
                    _save_failed_overlay(path, image_np)
                batch_rows[local_idx] = _make_row(rec, np.zeros(n_classes, dtype=np.int32))
                continue

            counts = _count_classes_from_result(result0, n_classes)
            if pred_dir is not None:
                try:
                    rendered = result0.plot()
                    _save_pred_image(path, rendered)
                except Exception as render_exc:  # noqa: BLE001
                    pred_failed_count += 1
                    _warn_unreadable(path, f"failed to render prediction image: {render_exc}")
                    _save_pred_image(path, image_np)

            batch_rows[local_idx] = _make_row(rec, counts)

        for local_idx, rec in enumerate(batch_recs):
            row = batch_rows[local_idx]
            if row is None:
                row = _make_row(rec, np.zeros(n_classes, dtype=np.int32))
            rows.append(row)

    if logger and unreadable_count > 0:
        logger.warning(
            "Unreadable images summary: %d files were zero-filled during inference.",
            unreadable_count,
        )
    if logger and batch_fallback_count > 0:
        logger.warning(
            "Batch fallback summary: %d batches required per-image retry.",
            batch_fallback_count,
        )
    if logger and pred_dir is not None:
        logger.info(
            "Saved prediction visualization images: %d (save failures: %d) | dir=%s",
            pred_saved_count,
            pred_failed_count,
            pred_dir,
        )

    columns = ["box", "seq", "image_path", "timestamp", "lat", "lon", "parse_ok", *class_names]
    return pd.DataFrame(rows, columns=columns)
