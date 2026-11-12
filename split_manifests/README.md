# Classification split manifest

`classification_split_manifest.csv` provides the image membership of the species- and family-level classification partitions. Photographs are not included in this file.

| Model level | Split | Images | Classes represented |
| --- | --- | ---: | ---: |
| Species | Train | 1,613 | 111 |
| Species | Validation | 375 | 111 |
| Family | Train | 1,576 | 39 |
| Family | Validation | 413 | 39 |
| Family | Test | 102 | 18 |

The CSV contains four columns:

- `model_level`: `species` or `family`;
- `split`: `train`, `val` or `test`;
- `class_name`: class-folder name;
- `image_filename`: photograph filename.

There is no independent species test split. The species validation partition must not be described as a test set. The strict 107-species workflow removes four higher-rank folders from the 111-class species partition and writes a run-specific manifest with the training outputs.
