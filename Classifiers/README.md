# Classification models

The two YOLO11-large classification checkpoints and their validation metrics are:

| File | Output classes | Top-1 | Top-5 | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `species_level.pt` | 107 | 0.78075 | 0.94118 | `ee7185d89502545dcf12cc4691c1b5833b707b6f57b1505ef5e9b2cef66f79e5` |
| `family_level.pt` | 39 | 0.89346 | 0.97337 | `150b4d1140dbae298557866293c75357e65b041c67ad06673626827958e7d8c4` |

`provenance/` contains the training arguments, epoch-level results, safely extracted checkpoint metadata, per-class metrics and per-image species validation predictions. Precision and recall for the 107-class species checkpoint were recalculated on its frozen 374-image validation partition using support-weighted averaging (precision 0.76007; recall 0.78075).

The species checkpoint contains 107 species-labelled classes. The number 107 describes the classification folders, rather than the complete COCO category vocabulary.

See [`../split_manifests/README.md`](../split_manifests/README.md) for the frozen image-level classification partition lists and their scope.
