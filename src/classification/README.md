# Species classification retraining

`retrain_species_107.sh` derives a 107-class dataset from the existing
`data_spp_seed/train` and `data_spp_seed/val` folders. It preserves the current
image memberships and excludes these four higher-rank folders:

- `BRZ-Bramidae`
- `CAX-Ariidae`
- `FLF-Cantherhines-spp`
- `FLY-Exocoetidae`

The derived dataset uses hard links, so the photographs are not duplicated.
The script writes the exact train/validation manifest with SHA-256 checksums,
checks that no byte-identical image crosses the split, trains YOLO11-large with
the archived `data_spp_seed` settings and evaluates the best checkpoint.
With the recovered split, the expected result is 1,595 training images and 369
validation images across 107 classes.

Run from the repository root:

```bash
chmod +x src/classification/retrain_species_107.sh
./src/classification/retrain_species_107.sh
```

If the dataset is stored elsewhere:

```bash
SOURCE_DATASET=/path/to/data_spp_seed \
MODEL=/path/to/yolo11l-cls.pt \
./src/classification/retrain_species_107.sh
```

The run folder contains `args.yaml`, `results.csv`, `weights/best.pt`, the
training log and `publication_evaluation/`. The latter contains the exact split
manifest, per-image predictions, Top-1 and Top-5 accuracy, support-weighted
precision and recall, the confusion matrix and the precision-recall plot.
