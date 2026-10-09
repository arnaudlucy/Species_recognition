# Annotations and supporting metadata for images of tropical pelagic species from French tuna fisheries observer programmes in the Atlantic and Indian Oceans

> **REDUCE Project** Reducing bycatch of threatened megafauna in the East Central Atlantic  
> EU Horizon Europe · Grant Agreement No. 101135583  
> **IDIL Graduate Program, University of Montpellier** · France 2030 · ANR-21-SFRI-0004

[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.13-blue)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11%20v8.3.78-orange)](https://github.com/ultralytics/ultralytics)

---

## Overview

This repository provides the data curation and preparation code accompanying the annotation dataset, together with the internship model workflows. Photographs and annotations are distributed through separate Zenodo deposits.

The dataset covers **107 species across 39 taxonomic families**, with **2,002 annotated images** and **2,887 instances**, labelled at both species and family level using polygon segmentation masks and bounding boxes.

- **Photographs:** [IRD-Ob7 photo library, version v2](https://zenodo.org/records/18788432), DOI [10.5281/zenodo.18788432](https://doi.org/10.5281/zenodo.18788432).
- **Annotations and supporting metadata:** [Zenodo annotation deposit](https://zenodo.org/records/23194303?preview=1), reserved DOI `10.5281/zenodo.23194303`. This deposit is currently a draft; access requires authorization until publication.

The annotations document the corpus used during the internship. The photo library is maintained independently; this annotation release is not automatically synchronized with later additions or filename changes.

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
│   ├── 01_data_curation.ipynb       ← Original internship notebook
│   ├── 02_data_preparation.ipynb    ← Corrected preparation workflow ← START HERE
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

Download photographs from the [IRD-Ob7 photo library, version v2](https://zenodo.org/records/18788432). Annotations and supporting metadata are in the [separate annotation deposit](https://zenodo.org/records/23194303?preview=1) (currently a draft).

Unzip the image archive into `data/images/`.  
Images are organised into sub-folders named with the FAO 3-alpha code (e.g. `BET-Thunnus-obesus/`).

### 4 · Run the preparation notebook

Install the CPU preparation dependencies:

```bash
python -m pip install -r requirements-preparation.txt
```

Open `notebooks/02_data_preparation.ipynb` and run all cells. Paths are resolved from the repository location. This notebook converts annotations, derives family labels, exports Hasty attributes and writes a segmentation split manifest. It does not train models or copy photographs.

Outputs are written to `data/prepared`, `data/prepared_families`, `data/prepared_attributes` and `data/prepared_split`. Source annotations are preserved. Identical outputs can be reused; choose a fresh output directory if parameters change. Multipart annotations are rejected rather than converted into multiple objects.

The original `notebooks/01_data_curation.ipynb` remains available with its original comments and experimental cells. Use `02_data_preparation.ipynb` for the corrected preparation workflow. Newly generated splits do not replace the historical splits underlying the reported model metrics.

Run the preparation checks with:

```bash
python -m unittest discover -s tests -v
```

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

Classification model files are provided in [`Classifiers/species_level.pt`](Classifiers/species_level.pt) and [`Classifiers/family_level.pt`](Classifiers/family_level.pt). Their presence is distinct from the availability of the historical image-level train/validation manifests.

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

Please cite the resources you use:

- Sabarros, P. S. and Lebranchu, J. (2026). *Photothèque des espèces rencontrées dans le cadre des programmes d’observation des pêcheries tropicales françaises (senne et palangre)*, v2. [Zenodo](https://doi.org/10.5281/zenodo.18788432).
- Arnaud, L., Restrepo-Ortiz, C., Sabarros, P. S., and Kaplan, D. M. *Annotations and supporting metadata for images of tropical pelagic species from French tuna fisheries observer programmes in the Atlantic and Indian Oceans*, version 1.0.0. [Zenodo draft](https://zenodo.org/records/23194303?preview=1), reserved DOI `10.5281/zenodo.23194303` (not yet registered).

---

## Funding

This work was supported by the REDUCE project (*Reducing bycatch of threatened megafauna in the East Central Atlantic*), Grant Agreement No. 101135583, funded by the European Commission through the HORIZON EUROPE Programme. Lucy Arnaud also received financial support from the IDIL Graduate Program of the University of Montpellier, funded by the French Government through France 2030 and the Agence nationale de la recherche (ANR-21-SFRI-0004).

---

## License

- **Code** — MIT License  
- **Data and annotations** — [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Photographs hosted in the associated IRD-Ob7 deposit are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); they are distributed separately from the annotations.
