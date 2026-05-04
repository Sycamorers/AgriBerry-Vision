# Processing Workflow

This repository stops at the data-processing stage used before QGIS visualization.

## Detection Processing

Each image is processed independently by YOLO. The detector output is converted into one per-image count vector containing the strawberry phenology classes `FL`, `G`, `W`, `P`, and `R`.

GPS metadata is parsed from the image filename. Records with invalid coordinates are excluded from GPS aggregation.

## GPS Aggregation

Images acquired at the same sampling location are grouped by rounded latitude and longitude. The default rounding precision is 6 decimal places. Category values are aggregated with one selected reducer:

- `mean`
- `sum`
- `max`

The default is `mean`, which produces one location-level value per class.

## Canopy Processing

The canopy branch uses a YOLO plant detector to propose plant boxes, then MobileSAM to segment each plant region. The exported `canopy` value is the aggregated mask pixel area at each GPS location.

## QGIS Export

After trial-level detection and canopy outputs are generated, `run_qgis_export.py` merges trials into box-level layers and writes:

- one GPS point layer,
- one CSV and GeoJSON layer per fruit class,
- one CSV and GeoJSON canopy layer,
- a box-level `box_agg_points.csv`,
- an index and run summary.

No final publication map is rendered by this repository. The final interpolation, classification, symbology, and layout are handled in QGIS.
