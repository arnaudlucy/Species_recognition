# Archived classification models

The two YOLO11-large classification checkpoints are preserved exactly as archived. Their SHA-256 checksums and checkpoint-stored validation metrics are:

| File | Output classes | Top-1 | Top-5 | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `species_level.pt` | 111 | 0.78153 | 0.94820 | `2e6a15c5991c7ad1e7cf6da8184bb671b173d834ff47fdc02a9f4466131ff7e7` |
| `family_level.pt` | 39 | 0.89346 | 0.97337 | `150b4d1140dbae298557866293c75357e65b041c67ad06673626827958e7d8c4` |

`provenance/` contains the training arguments, epoch-level results and safely extracted checkpoint metadata. These files document the archived models; no model was retrained during repository preparation.

The 111 species-level outputs comprise 107 species-labelled classes and four higher-rank groups: `BRZ-Bramidae`, `CAX-Ariidae`, `FLF-Cantherhines-spp` and `FLY-Exocoetidae`. The number 107 therefore describes the classification folders, rather than the complete COCO category vocabulary.

See [`../split_manifests/README.md`](../split_manifests/README.md) for the recovered image lists and their evidence limitations.
