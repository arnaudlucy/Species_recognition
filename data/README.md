# Data and preparation

## Files in this repository

| File | Contents |
|---|---|
| [`final_data.json`](final_data.json) | Original COCO annotations exported from Hasty, with the original image and category IDs. |
| [`additional_images_informations.json`](additional_images_informations.json) | Hasty export of biological and contextual attributes recorded during annotation. |

## Photographs and supporting data

Download available photographs from the [IRD-Ob7 photographic collection, version v2](https://doi.org/10.5281/zenodo.18788432) and place them under `data/images/`. Match photographs using the COCO `images.file_name` field and report unmatched filenames. Complete coverage of the annotated corpus by the independently maintained photographic release has not been verified.

The [annotation deposit](https://doi.org/10.5281/zenodo.23194303) provides annotations and supporting data. See the [main README](../README.md) for setup and model downloads.

## Outputs generated locally

Run [`notebooks/02_data_preparation.ipynb`](../notebooks/02_data_preparation.ipynb) with the CPU preparation dependencies. The notebook creates these outputs under `data/`:

| Directory | Contents |
|---|---|
| `prepared/` | YOLO labels and class mapping derived from the COCO export. |
| `prepared_families/` | Family-level labels and class mapping. |
| `prepared_attributes/` | Attribute tables and matching report. |
| `prepared_split/` | Newly generated segmentation split manifest, when the split step is enabled. |

For the classification results reported with the released weights, use the [frozen classification manifest](../split_manifests/classification_split_manifest.csv).
