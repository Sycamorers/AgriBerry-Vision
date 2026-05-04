from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_QGIS_STEM = "gps_points_qgis"
DEFAULT_DISTRIBUTION_DIRNAME = "qgis_distribution_layers"
_AGG_BASE_COLUMNS = {"box", "seq", "lat", "lon", "n_images_at_point"}


def _normalize_qgis_points_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty or "lon" not in agg_df.columns or "lat" not in agg_df.columns:
        return pd.DataFrame(columns=["longitude", "latitude"])

    points_df = agg_df.loc[:, ["lon", "lat"]].copy()
    points_df = points_df.rename(columns={"lon": "longitude", "lat": "latitude"})
    points_df = points_df.dropna(subset=["longitude", "latitude"])
    if points_df.empty:
        return pd.DataFrame(columns=["longitude", "latitude"])

    points_df["longitude"] = points_df["longitude"].astype(float)
    points_df["latitude"] = points_df["latitude"].astype(float)
    points_df = points_df.drop_duplicates(subset=["longitude", "latitude"], keep="first").reset_index(drop=True)
    return points_df[["longitude", "latitude"]]


def write_qgis_points_from_agg_df(
    agg_df: pd.DataFrame,
    output_dir: Path | str,
    stem: str = DEFAULT_QGIS_STEM,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    points_df = _normalize_qgis_points_df(agg_df)
    csv_path = output_dir_path / f"{stem}.csv"
    geojson_path = output_dir_path / f"{stem}.geojson"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["longitude", "latitude"])
        writer.writeheader()
        for row in points_df.to_dict(orient="records"):
            writer.writerow(row)

    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": {},
            }
            for row in points_df.to_dict(orient="records")
        ],
    }
    geojson_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "point_count": int(len(points_df)),
        "csv_path": str(csv_path.resolve()),
        "geojson_path": str(geojson_path.resolve()),
        "crs": "EPSG:4326",
    }


def infer_distribution_value_columns(agg_df: pd.DataFrame) -> list[str]:
    return [column for column in agg_df.columns if column not in _AGG_BASE_COLUMNS]


def _normalize_distribution_df(agg_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if agg_df.empty or "lon" not in agg_df.columns or "lat" not in agg_df.columns or value_column not in agg_df.columns:
        return pd.DataFrame(columns=["longitude", "latitude", "value"])

    out_df = agg_df.loc[:, ["lon", "lat", value_column]].copy()
    out_df = out_df.rename(columns={"lon": "longitude", "lat": "latitude", value_column: "value"})
    out_df = out_df.dropna(subset=["longitude", "latitude", "value"])
    if out_df.empty:
        return pd.DataFrame(columns=["longitude", "latitude", "value"])

    out_df["longitude"] = out_df["longitude"].astype(float)
    out_df["latitude"] = out_df["latitude"].astype(float)
    out_df["value"] = out_df["value"].astype(float)
    out_df = out_df[["longitude", "latitude", "value"]].reset_index(drop=True)
    return out_df


def _write_distribution_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["longitude", "latitude", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_distribution_geojson(path: Path, rows: list[dict[str, float]]) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": {
                    "value": float(row["value"]),
                },
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_distribution_layers_from_agg_df(
    agg_df: pd.DataFrame,
    output_dir: Path | str,
    dirname: str = DEFAULT_DISTRIBUTION_DIRNAME,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir) / dirname
    output_dir_path.mkdir(parents=True, exist_ok=True)

    value_columns = infer_distribution_value_columns(agg_df)
    target_summaries: list[dict[str, Any]] = []

    for value_column in value_columns:
        layer_df = _normalize_distribution_df(agg_df=agg_df, value_column=value_column)
        rows = layer_df.to_dict(orient="records")

        csv_path = output_dir_path / f"{value_column}.csv"
        geojson_path = output_dir_path / f"{value_column}.geojson"
        _write_distribution_csv(csv_path, rows=rows)
        _write_distribution_geojson(geojson_path, rows=rows)

        target_summaries.append(
            {
                "target": value_column,
                "point_count": int(len(layer_df)),
                "csv_path": str(csv_path.resolve()),
                "geojson_path": str(geojson_path.resolve()),
            }
        )

    manifest_path = output_dir_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "point_count", "csv_path", "geojson_path"])
        writer.writeheader()
        writer.writerows(target_summaries)

    return {
        "target_count": int(len(target_summaries)),
        "targets": target_summaries,
        "manifest_path": str(manifest_path.resolve()),
        "crs": "EPSG:4326",
    }
