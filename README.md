# Annotations and supporting metadata for images of tropical pelagic species from French tuna fisheries observer programs in the Atlantic and Indian Oceans

> **REDUCE Project** Reducing bycatch of threatened megafauna in the East Central Atlantic  
> EU Horizon Europe Grant Agreement No. 101135583  
> **IDIL Graduate Program, University of Montpellier**  France 2030  ANR-21-SFRI-0004

[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.13-blue)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11%20v8.3.78-orange)](https://github.com/ultralytics/ultralytics)

---

## Overview

This repository provides the data curation and preparation code accompanying the annotation dataset, together with the model workflows. Photographs and annotations are distributed through separate Zenodo deposits.

The COCO file contains **2,002 image records** and **2,887 annotated instances**. An instance is one individual delineated by a polygon segmentation mask and a bounding box. Of 125 declared annotation categories, 123 are populated: 110 carry species names and 13 are unresolved `Unknown` groups. The frozen classification benchmarks contain **107 species-labelled classes** and **39 family-level classes**. These classification counts and the COCO category vocabulary are related but are not interchangeable.

- **Photographs:** [IRD-Ob7 photo library, version v2](https://zenodo.org/records/18788432), DOI [10.5281/zenodo.18788432](https://doi.org/10.5281/zenodo.18788432).
- **Annotations and supporting metadata:** [Zenodo annotation deposit](https://zenodo.org/records/23194303), DOI `10.5281/zenodo.23194303`. 

The annotations document the corpus used during the internship. The photo library is maintained independently; this annotation release is not automatically synchronized with later additions or filename changes.

---

## Repository Structure

```
Species_recognition/
│
├── README.md                        ← This file
├── LICENSE                          ← CC BY 4.0
├── environment.yml                  ← Conda environment (pinned versions)
├── split_manifests/                 ← Image-level classification partition list
├── Classifiers/
│   ├── species_level.pt             ← Validated 107-class checkpoint
│   ├── family_level.pt              ← 39-class checkpoint
│   └── provenance/                  ← Arguments, epoch results and checkpoint metadata
│
├── notebooks/
│   ├── 01_data_curation.ipynb       ← Original internship notebook
│   ├── 02_data_preparation.ipynb    ← Corrected preparation workflow ← START HERE
│   ├── ASFIS_sp_2025.csv            ← FAO ASFIS species reference
│   ├── metadata_phototheque.csv     ← Per-image metadata
│   └── species_ecology.csv          ← Taxonomic and ecological info, used for family grouping
│
├── data/
│   ├── README.md                    ← Explains what goes here and where to get images
│   ├── final_data.json              ← Raw COCO annotation export from Hasty.ai
│   └── images/                      ← Download photographs separately
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

Download photographs from the [IRD-Ob7 photo library, version v2](https://zenodo.org/records/18788432). Annotations and supporting metadata are in the [separate annotation deposit](https://zenodo.org/records/23194303).

Unzip the image archive into `data/images/`.  
Images are organized into sub-folders named with the FAO 3-alpha code (e.g., `BET-Thunnus-obesus/`).

### 4 · Run the preparation notebook

Install the CPU preparation dependencies:

```bash
python -m pip install -r requirements-preparation.txt
```

Open `notebooks/02_data_preparation.ipynb` and run all cells. Paths are resolved from the repository location. This notebook converts annotations, derives family labels, exports Hasty attributes and writes a segmentation split manifest. It does not train models or copy photographs.

Outputs are written to `data/prepared`, `data/prepared_families`, `data/prepared_attributes` and `data/prepared_split`. Source annotations are preserved. You can reuse identical outputs; choose a fresh output directory if parameters change. Multipart annotations are rejected rather than converted into multiple objects.

The original `notebooks/01_data_curation.ipynb` remains available with its original comments and experimental cells. Use `02_data_preparation.ipynb` for the corrected preparation workflow. Newly generated splits do not replace the historical splits underlying the reported model metrics.

Run the preparation checks with:

```bash
python -m unittest discover -s tests -v
```

## Dataset Summary

| COCO annotation vocabulary | Count |
|---|---:|
| Images | 2,002 |
| Annotated instances | 2,887 |
| Declared categories | 125 |
| Populated categories | 123 |
| Populated species-named categories | 110 |
| Populated unresolved `Unknown` groups | 13 |

The current classification manifest contains 107 species-labelled classes and 39 family classes. See [`split_manifests/README.md`](split_manifests/README.md) for the frozen image-level train, validation and family test memberships.

---

## Model Performance (YOLO11-large)

Classification model files are provided in [`Classifiers/species_level.pt`](Classifiers/species_level.pt) and [`Classifiers/family_level.pt`](Classifiers/family_level.pt). The species values were recalculated on the frozen validation partition; the family values are recorded in the checkpoint. Both are validation metrics.

| Level | Output classes | Top-1 accuracy | Top-5 accuracy |
|------|---:|---:|---:|
| Species | 107 | 78.075% | 94.118% |
| Family | 39 | 89.346% | 97.337% |

Training arguments, epoch results, extracted checkpoint metadata, per-image validation predictions and SHA-256 checksums are documented in [`Classifiers/README.md`](Classifiers/README.md). The exact 107-class species split contains 1,595 training images and 374 validation images, with no byte-identical file crossing the split.

---

## Citation

How to cite:

- Sabarros, P. S. and Lebranchu, J. (2026). *Photothèque des espèces rencontrées dans le cadre des programmes d’observation des pêcheries tropicales françaises (senne et palangre)*, v2. [Zenodo](https://doi.org/10.5281/zenodo.18788432).
- Arnaud, L., Restrepo-Ortiz, C., Sabarros, P. S., and Kaplan, D. M. *Annotations and supporting metadata for images of tropical pelagic species from French tuna fisheries observer programs in the Atlantic and Indian Oceans*, version 1.0.0. [Zenodo](https://zenodo.org/records/23194303), DOI `10.5281/zenodo.23194303`.

---

## Funding

This work was supported by the REDUCE project (*Reducing bycatch of threatened megafauna in the East Central Atlantic*), Grant Agreement No. 101135583, funded by the European Commission through the HORIZON EUROPE Programme. Lucy Arnaud also received financial support from the IDIL Graduate Program of the University of Montpellier, funded by the French Government through France 2030 and the Agence nationale de la recherche (ANR-21-SFRI-0004).

---

## License

- **Code**  MIT License  
- **Data and annotations**  [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Photographs hosted in the associated IRD-Ob7 deposit are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); they are distributed separately from the annotations.
