from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class GPSParseResult:
    timestamp: str
    lat: float
    lon: float
    parse_ok: bool
    error: str = ""


def _parse_float_token(token: str) -> float:
    """Parse float token directly; if needed, extract embedded numeric substring."""
    try:
        return float(token)
    except (TypeError, ValueError):
        match = _FLOAT_RE.search(str(token))
        if match is None:
            raise
        return float(match.group(0))


def parse_gps_from_stem(stem: str) -> GPSParseResult:
    """Parse <timestamp>_<lat>_<lon> from filename stem.

    Parsing behavior follows required token-based logic and includes a tolerant numeric
    extraction fallback for tokens like ``lat29.4`` / ``lon-82.1``.
    """
    tokens = stem.split("_")
    if len(tokens) < 3:
        return GPSParseResult(
            timestamp="",
            lat=0.0,
            lon=0.0,
            parse_ok=False,
            error="expected at least 3 underscore-separated tokens",
        )

    timestamp = "_".join(tokens[:-2])
    lat_token = tokens[-2]
    lon_token = tokens[-1]

    try:
        lat = _parse_float_token(lat_token)
        lon = _parse_float_token(lon_token)
    except Exception as exc:  # noqa: BLE001
        return GPSParseResult(
            timestamp=timestamp,
            lat=0.0,
            lon=0.0,
            parse_ok=False,
            error=f"failed to parse lat/lon tokens ({lat_token}, {lon_token}): {exc}",
        )

    # Safety correction: if values are impossible as (lat, lon) but valid when swapped, swap.
    if (abs(lat) > 90 or abs(lon) > 180) and (abs(lon) <= 90 and abs(lat) <= 180):
        lat, lon = lon, lat

    if abs(lat) > 90 or abs(lon) > 180:
        return GPSParseResult(
            timestamp=timestamp,
            lat=0.0,
            lon=0.0,
            parse_ok=False,
            error=f"parsed coordinates out of range: lat={lat}, lon={lon}",
        )

    return GPSParseResult(timestamp=timestamp, lat=lat, lon=lon, parse_ok=True, error="")


def parse_image_path(image_path: Path | str) -> GPSParseResult:
    """Parse GPS metadata from image filename stem."""
    path = Path(image_path)
    return parse_gps_from_stem(path.stem)
