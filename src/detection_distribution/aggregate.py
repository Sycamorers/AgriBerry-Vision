from __future__ import annotations

from typing import List

import pandas as pd

_BASE_COLUMNS = {
    "box",
    "seq",
    "image_path",
    "timestamp",
    "lat",
    "lon",
    "parse_ok",
}


def infer_class_columns(df: pd.DataFrame) -> List[str]:
    """Infer class count columns from detection DataFrame."""
    return [c for c in df.columns if c not in _BASE_COLUMNS]


def aggregate_points(
    detections_df: pd.DataFrame,
    agg: str = "mean",
    gps_round: int = 6,
) -> pd.DataFrame:
    """Aggregate per-image detections onto rounded GPS points."""
    if agg not in {"mean", "sum", "max"}:
        raise ValueError(f"Unsupported agg: {agg}")

    class_cols = infer_class_columns(detections_df)

    out_columns = ["box", "seq", "lat", "lon", "n_images_at_point", *class_cols]
    if detections_df.empty:
        return pd.DataFrame(columns=out_columns)

    work_df = detections_df.copy()
    work_df["lat_round"] = work_df["lat"].astype(float).round(gps_round)
    work_df["lon_round"] = work_df["lon"].astype(float).round(gps_round)

    group_cols = ["box", "seq", "lat_round", "lon_round"]
    grouped = work_df.groupby(group_cols, as_index=False)

    size_df = grouped.size().rename(columns={"size": "n_images_at_point"})

    if class_cols:
        class_agg_df = grouped[class_cols].agg(agg)
        agg_df = size_df.merge(class_agg_df, on=group_cols, how="left")
    else:
        agg_df = size_df.copy()

    agg_df = agg_df.rename(columns={"lat_round": "lat", "lon_round": "lon"})
    agg_df = agg_df[["box", "seq", "lat", "lon", "n_images_at_point", *class_cols]]

    if agg in {"sum", "max"} and class_cols:
        for col in class_cols:
            agg_df[col] = agg_df[col].round().astype(int)

    agg_df["n_images_at_point"] = agg_df["n_images_at_point"].astype(int)
    agg_df = agg_df.sort_values(["lat", "lon"], ascending=[True, True]).reset_index(drop=True)
    return agg_df
