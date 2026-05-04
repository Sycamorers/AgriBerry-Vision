# QGIS Outputs

`run_qgis_export.py` writes final layers under `QGIS_out/{date}/{box}/`.

## Layer Files

| File | Contents |
| --- | --- |
| `gps_points_qgis.csv` | Unique GPS sampling points as longitude/latitude |
| `gps_points_qgis.geojson` | Same GPS points as GeoJSON |
| `FL.csv`, `G.csv`, `W.csv`, `P.csv`, `R.csv` | Per-class point values |
| `FL.geojson`, `G.geojson`, `W.geojson`, `P.geojson`, `R.geojson` | Per-class point values as GeoJSON |
| `canopy.csv` | Canopy-size point values |
| `canopy.geojson` | Canopy-size point values as GeoJSON |
| `box_agg_points.csv` | Combined table with all exported variables |
| `manifest.csv` | Inventory of exported target layers |

## Coordinate Reference System

The exported coordinates are WGS84 longitude/latitude (`EPSG:4326`).

## Recommended QGIS Use

Load the GeoJSON files directly, or load the CSV files with `longitude` as X and `latitude` as Y. Use QGIS-native interpolation, raster styling, and print layout tools for final figures.
