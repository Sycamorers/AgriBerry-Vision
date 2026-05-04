# AgriBerry Vision

Data-processing code for the paper:

**AgriBerry Vision: A Sprayer-Mounted Georeferenced Imaging Module for Strawberry Phenology Monitoring**

This repository contains the processing stage only. Raw sprayer-mounted field images are converted into georeferenced detection and canopy layers, and the final map rendering is performed in QGIS rather than by the earlier Python plotting workflow.

## Scope

The pipeline:

1. Scans a `data/{date}/{box}/{trial}/` image hierarchy.
2. Parses timestamp, latitude, and longitude from image filenames.
3. Runs YOLO-based strawberry phenology detection.
4. Aggregates repeated images at rounded GPS locations.
5. Runs plant detection plus MobileSAM canopy segmentation.
6. Exports box-level CSV and GeoJSON layers for QGIS.

The repository intentionally excludes raw images, model checkpoints, generated QGIS outputs, and old Python-rendered heatmap figures.

## Dataset

The raw image dataset is hosted separately on Hugging Face:

https://huggingface.co/datasets/Sycamorers/ABV

The uploaded archive is available at:

https://huggingface.co/datasets/Sycamorers/ABV/blob/main/abv_raw_images_data.tar.gz

Current local dataset summary:

| Date | Box | Trial | Images |
| --- | --- | --- | ---: |
| 260316 | box1 | dc_4 | 763 |
| 260316 | box2 | dc_4 | 866 |
| 260316 | box3 | dc_4 | 609 |
| 260324 | box1 | dc_1 | 976 |
| 260324 | box2 | dc_1 | 721 |
| 260324 | box3 | dc_1 | 711 |

Total: 4,646 JPG images, approximately 2.54 GB before compression.

## Input Layout

```text
data/
  260316/
    box1/
      dc_4/
        2026_01_15_17-29-09-934_lat29.40424481333333_lon-82.1418702.jpg
```

Filenames must include parseable `lat...` and `lon...` tokens. Images with invalid GPS metadata are skipped from spatial aggregation.

## Classes

The fruit detector is expected to expose these strawberry phenology classes:

| Class | Meaning |
| --- | --- |
| FL | Flower |
| G | Green fruit |
| W | White fruit |
| P | Pink fruit |
| R | Red fruit |

The canopy pipeline exports an additional `canopy` layer derived from plant detection and MobileSAM mask area.

## Setup

```bash
conda create -n abv python=3.11
conda activate abv
pip install -r requirements.txt
```

Model checkpoints are not tracked in Git. Place them under `ckpt/`:

```text
ckpt/
  fruit.pt
  plant.pt
  mobile_sam.pt
```

## Run

```bash
python run_qgis_export.py \
  --input-root data \
  --output-root QGIS_out \
  --weights ckpt/fruit.pt \
  --detector-weights ckpt/plant.pt \
  --sam-checkpoint ckpt/mobile_sam.pt \
  --device 0 \
  --overwrite
```

Use `--device cpu` if GPU inference is unavailable.

## Outputs

The main QGIS output structure is:

```text
QGIS_out/
  index.csv
  run_summary.json
  {date}/
    {box}/
      gps_points_qgis.csv
      gps_points_qgis.geojson
      box_agg_points.csv
      FL.csv / FL.geojson
      G.csv / G.geojson
      W.csv / W.geojson
      P.csv / P.geojson
      R.csv / R.geojson
      canopy.csv / canopy.geojson
      manifest.csv
```

All exported point layers use longitude/latitude coordinates in EPSG:4326. Load the CSV or GeoJSON layers in QGIS, then apply the interpolation and symbology used for the final figures.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `run_qgis_export.py` | Main end-to-end processor for QGIS-ready layers |
| `run_detection_distribution.py` | Detection-only trial processor and QGIS export helper |
| `src/detection_distribution/` | GPS parsing, image discovery, YOLO inference, aggregation, and layer export |
| `src/canopy_distribution/` | Plant detection, MobileSAM segmentation, canopy aggregation |
| `docs/` | Processing and QGIS workflow notes |

## Citation

If this code or dataset supports your work, cite the accompanying manuscript:

AgriBerry Vision: A Sprayer-Mounted Georeferenced Imaging Module for Strawberry Phenology Monitoring.
