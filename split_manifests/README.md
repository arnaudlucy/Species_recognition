# Recovered classification split manifests

These CSV files list image membership recovered from the MARBEC storage folders on 5 November 2026. They contain file names, relative paths, class folders and file sizes; photographs are not included.

| Manifest | Images | Classes | Evidence |
| --- | ---: | ---: | --- |
| `species_seed_train.csv` | 1,613 | 111 | Recovered from `data_spp_seed/train` |
| `species_seed_val.csv` | 375 | 111 | Recovered from `data_spp_seed/val` |
| `family_train.csv` | 1,576 | 39 | Recovered from `dataset_split_family/data/train` |
| `family_val.csv` | 413 | 39 | Recovered from `dataset_split_family/data/val` |
| `family_test.csv` | 102 | 18 | Recovered from `dataset_split_family/data/test` |

The family validation class counts match the archived family prediction table and the family checkpoint points to `dataset_split_family/data`. The family manifests can therefore be linked to the included family model with documented evidence.

The species manifests have the same 111 class-folder names as the included species checkpoint. An accessible training log also links `data_spp_seed` to a separate run with 1,613 training and 375 validation images, but that run directory is now empty. The included species checkpoint records a different run name and an unavailable path named `dataset`. Consequently, these species manifests are published as the closest recovered split candidate, not as proven image-level membership for `Classifiers/species_level.pt`.

There is no recovered independent species test split. The validation split must not be described as an independent test set.
