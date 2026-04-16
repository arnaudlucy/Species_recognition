# Species Recognition in Tropical Tuna Fisheries

> **REDUCE Project** — Reducing bycatch of threatened megafauna in the East Central Atlantic  
> EU Horizon Europe · Grant Agreement No. 101135583

[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.13-blue)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11%20v8.3.78-orange)](https://github.com/ultralytics/ultralytics)

---

## Overview

This repository contains the data curation pipeline and reproducibility code for the annotated dataset of bycatch species from French tropical tuna fisheries described in:

> **Arnaud L et al.** *(2025)*. *An annotated image dataset of epipelagic bycatch species from French tropical tuna fisheries observer programs.* Data in Brief (submitted).

The dataset covers **107 species across 39 taxonomic families**, with **2,002 annotated images** and **2,887 instances**, labelled at both species and family level using polygon segmentation masks and bounding boxes.

Images are available on **Zenodo**: `[DOI to be added upon publication]`

---

## Repository Structure

```
Species_recognition/
│
├── README.md                        ← This file
├── LICENSE                          ← CC BY 4.0
├── environment.yml                  ← Conda environment (pinned versions)
│
├── notebooks/
│   ├── 01_data_curation.ipynb       ← COCO → YOLO conversion pipeline  ← START HERE
│   ├── metadata_phototeque.csv      ← Per-image metadata
│   └── species_ecology.csv          ← Taxonomic and ecological info, used for family grouping
│
├── data/
│   ├── README.md                    ← Explains what goes here and where to get images
│   ├── final_data.json              ← Raw COCO annotation export from Hasty.ai
│   ├── dataset_coco_reindexed.json  ← COCO with 0-indexed categories (output of cell 3)
│   ├── classes.txt                  ← Class ID → name mapping (output of cell 6)
│   ├── classes_final.txt            ← After rare-species merging (output of cell 17)
│   ├── annotations.csv              ← Annotation metadata CSV (output of cell 7)
│   ├── images/                      ←   Download from Zenodo and unzip here
│   │   └── <FAO_code>-<Genus>-<species>/
│   │       └── *.jpg
│   ├── processed_labels/            ← YOLO .txt label files (output of cell 7)
│   └── dataset/                     ← Train/val split (output of cell 14)
│       ├── images/train/
│       ├── images/val/
│       ├── labels/train/
│       └── labels/val/
│
└── results/
    ├── summary_families_photo_counts.csv
    └── taxonomy_summary_by_category.csv
```

---

## Quickstart

### 1 · Clone the repository

```bash
git clone https://github.com/arnaudlucy/Species_recognition.git
cd Species_recognition
```

### 2 · Set up the environment

```bash
conda env create -f environment.yml
conda activate species_recognition
```

### 3 · Download the images from Zenodo

```
[Zenodo DOI link — to be added upon publication]
```

Unzip the image archive into `data/images/`.  
Images are organised into sub-folders named with the FAO 3-alpha code (e.g. `BET-Thunnus-obesus/`).

### 4 · Run the curation notebook

Open and run **`notebooks/01_data_curation.ipynb`** from top to bottom.  
All paths are configured in **Cell 2** — you should not need to change anything else.

| Step | Cell | Output |
|------|------|--------|
| Configure paths | 2 | — |
| Reindex COCO categories | 3 | `data/dataset_coco_reindexed.json` |
| Load & validate JSON | 4–5 | — |
| Extract class list | 6 | `data/classes.txt` |
| Convert COCO → YOLO | 7 | `data/processed_labels/` + `data/annotations.csv` |
| Visual QC | 8–11 | — |
| Annotation counts & plots | 12–13 | — |
| Multilabel stratified split (80/20) | 14 | `data/dataset/` |
| Simple per-folder split (alternative) | 15 | `data/dataset/` |
| Orphan label check | 16 | `data/labels_orphans.txt` |
| Rare-species merging | 17 | `data/classes_final.txt` |

---

## Dataset Summary

| Category | Families | Species | Images | Instances |
|----------|----------|---------|--------|-----------|
| Elasmobranchii | 13 | 31 | — | — |
| Teleostei | 18 | 53 | — | — |
| Reptilia / Mammalia / Other | 8 | 23 | — | — |
| **Total** | **39** | **107** | **2,002** | **2,887** |

See `results/taxonomy_summary_by_category.csv` for full details.

---

## Model Performance (YOLO11-large)

| Task | Level | Top-1 Accuracy | Precision | Recall |
|------|-------|---------------|-----------|--------|
| Classification | Species | 73.6% | 93.7% | 88.4% |
| Classification | Family  | 89.3% | 86.8% | 88.1% |
| Object Detection | Species | — | 84.6% (mAP@50: 61.8%) | 46.1% |
| Object Detection | Family  | — | 93.8% (mAP@50: 88.7%) | 81.4% |
| Instance Segmentation | Species | — | 72.3% (mAP@50: 71.8%) | 63.6% |
| Instance Segmentation | Family  | — | 91.4% (mAP@50: 83.2%) | 73.1% |

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{arnaud2025reduce,
  title   = {An annotated image dataset of epipelagic bycatch species
             from French tropical tuna fisheries observer programs},
  author  = {Arnaud, Lucy Zoe Marylou and Kaplan, David M. and
             Restrepo-Ortiz, Claudia},
  journal = {Data in Brief},
  year    = {2025},
  note    = {Submitted}
}
```

---

## Funding

This work was supported by the REDUCE project (*Reducing bycatch of threatened megafauna in the East Central Atlantic*), Grant Agreement No. 101135583, funded by the European Commission through the HORIZON EUROPE Programme.

---

## License

- **Code** — MIT License  
- **Data and annotations** — [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
