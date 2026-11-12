# Data-description figures

These figures are generated from `data/final_data.json` by running:

```bash
python src/figures/make_dataset_figures.py
```

## Category-instance distribution

`category_instance_distribution` shows the number of annotated instances in each populated COCO taxonomic category on a logarithmic scale. The released COCO file contains 125 declared categories, 123 populated categories and 2,887 annotated instances. An annotated instance is one individual delineated by a segmentation mask and its bounding box.

Suggested caption: **Distribution of annotated instances across populated taxonomic categories.** Categories are ranked by decreasing instance count and the vertical axis uses a logarithmic scale. The COCO vocabulary contains 125 declared categories, of which 123 are populated; two categories have no annotations.

## Image-selection flow

`image_selection_flow` documents the reported passage from 2,564 selected photographs to the 2,002 images in the released COCO export and its 2,887 annotated instances. The difference of 562 photographs is arithmetic; the available records do not distinguish uncompleted annotations from other exclusions.

Suggested caption: **Selection and annotation flow for the released dataset.** Of 2,564 photographs reported as selected for annotation, 2,002 are represented in the released COCO image table and contain 2,887 annotated instances. The available records do not distinguish uncompleted annotations from other exclusions among the remaining 562 photographs.

The PNG files are provided at 300 dpi. PDF and SVG versions are included for manuscript layout. `category_instance_counts.csv` contains the source counts for all declared categories.
