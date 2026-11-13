# Classification split manifest

`classification_split_manifest.csv` provides the image membership of the current species- and family-level classification partitions. Photographs are not included in this file.

| Model level | Split | Images | Classes represented |
| --- | --- | ---: | ---: |
| Species | Train | 1,595 | 107 |
| Species | Validation | 374 | 107 |
| Family | Train | 1,576 | 39 |
| Family | Validation | 413 | 39 |
| Family | Test | 102 | 18 |

The CSV contains four columns:

- `model_level`: `species` or `family`;
- `split`: `train`, `val` or `test`;
- `class_name`: class-folder name;
- `image_filename`: photograph filename.

There is no independent species test split. The species validation partition must not be described as a test set.

`species_107_split_manifest.csv` is the richer run-specific manifest for the validated species checkpoint. It also records relative paths, file sizes, source and derived SHA-256 hashes, and the preparation action. `classification_split_manifest_historical_111.csv` preserves the superseded 111-folder partition for provenance; it must not be associated with the current 107-class checkpoint.
