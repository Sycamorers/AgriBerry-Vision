# Data Folder

Place the raw AgriBerry Vision image hierarchy here when running the pipeline locally.

The dataset is hosted separately on Hugging Face:

https://huggingface.co/datasets/Sycamorers/ABV

Download and extract `abv_raw_images_data.tar.gz` from that dataset page to restore the expected local `data/` hierarchy.

Expected structure:

```text
data/{date}/{box}/{trial}/*.jpg
```
