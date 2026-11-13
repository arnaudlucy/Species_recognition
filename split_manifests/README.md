# Classification split manifest

`classification_split_manifest.csv` provides the image membership of the current species- and family-level classification partitions. Photographs are not included in this file.

| Model level | Split | Images | Classes represented |
| --- | --- | ---: | ---: |
| Species | Train | 1,595 | 107 |
| Species | Validation | 374 | 107 |
| Family | Train | 1,576 | 39 |
| Family | Validation | 413 | 39 |
| Family | Test | 102 | 18 |

The CSV contains the following columns:

- `model_level`: `species` or `family`;
- `split`: `train`, `val` or `test`;
- `class_name`: class-folder name;
- `image_filename`: photograph filename.
- `relative_path`, `size_bytes`, `source_sha256`, `sha256` and `preparation_action`: additional details for the species run. These fields are blank for family images where the information was not collected.

There is no independent species test split. The species validation partition must not be described as a test set.

