from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXCLUDED_CLASSES = {
    "BRZ-Bramidae",
    "CAX-Ariidae",
    "FLF-Cantherhines-spp",
    "FLY-Exocoetidae",
}
FAMILY_OVERRIDES = {
    "FIW_Balaenoptera-physalus": "Balaenopteridae",
    "PLS-Dasyatis-violacea": "Dasyatidae",
    "RMB-Manta-birostris": "Mobulidae",
    "SFA-Istiophorus-platypterus": "Istiophoridae",
    "TRG-Balistes-capriscus": "Balistidae",
}


def normalized_name(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def scientific_name_from_label(label):
    return re.sub(r"^[A-Za-z0-9]+[-_]", "", label, count=1)


def load_family_mapping(path, labels):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_name = {
        normalized_name(row["species_scientific_name"]): row["Family"]
        for row in rows
        if row.get("species_scientific_name") and row.get("Family")
    }
    mapping = {}
    missing = []
    for label in labels:
        family = FAMILY_OVERRIDES.get(label)
        if family is None:
            family = by_name.get(normalized_name(scientific_name_from_label(label)))
        if not family:
            missing.append(label)
        else:
            mapping[label] = family
    if missing:
        raise ValueError("Missing family mapping for: " + ", ".join(missing))
    return mapping


def class_directories(path):
    return sorted(directory for directory in Path(path).iterdir() if directory.is_dir())


def image_files(directory):
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith("._")
    )


def checkpoint_names(model):
    names = model.names
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def evaluate(model_path, data_path, taxonomy_path, output_path, imgsz=224, batch=32):
    model_path = Path(model_path).resolve()
    data_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=False)

    model = YOLO(str(model_path), task="classify")
    labels = checkpoint_names(model)
    if len(labels) != 107:
        raise ValueError(f"Expected a 107-class checkpoint, found {len(labels)} classes")
    unexpected = EXCLUDED_CLASSES.intersection(labels)
    if unexpected:
        raise ValueError("Excluded classes remain in the checkpoint: " + ", ".join(sorted(unexpected)))

    train_classes = {directory.name for directory in class_directories(data_path / "train")}
    val_classes = {directory.name for directory in class_directories(data_path / "val")}
    label_set = set(labels)
    if train_classes != label_set or val_classes != label_set:
        raise ValueError("Checkpoint, train and val class lists are not identical")

    family_by_class = load_family_mapping(taxonomy_path, labels)
    training_counts = {
        directory.name: len(image_files(directory))
        for directory in class_directories(data_path / "train")
    }

    validation_paths = []
    true_labels = []
    for directory in class_directories(data_path / "val"):
        files = image_files(directory)
        validation_paths.extend(files)
        true_labels.extend([directory.name] * len(files))

    predicted_labels = []
    top5_correct = []
    top1_confidence = []
    results = model.predict(
        source=[str(path) for path in validation_paths],
        imgsz=imgsz,
        batch=batch,
        stream=True,
        verbose=False,
    )
    for result, true_label in zip(results, true_labels):
        predicted = labels[int(result.probs.top1)]
        top5 = [labels[int(index)] for index in result.probs.top5]
        predicted_labels.append(predicted)
        top5_correct.append(true_label in top5)
        top1_confidence.append(float(result.probs.top1conf.item()))

    if len(predicted_labels) != len(validation_paths):
        raise RuntimeError("Prediction count does not match validation image count")

    predictions = pd.DataFrame(
        {
            "relative_path": [path.relative_to(data_path).as_posix() for path in validation_paths],
            "file_name": [path.name for path in validation_paths],
            "true_class": true_labels,
            "predicted_class": predicted_labels,
            "top1_correct": [a == b for a, b in zip(true_labels, predicted_labels)],
            "top5_correct": top5_correct,
            "top1_confidence": top1_confidence,
        }
    )
    predictions.to_csv(output_path / "validation_predictions.csv", index=False)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )
    predicted_counts = Counter(predicted_labels)
    true_positive = Counter(
        true_label
        for true_label, predicted_label in zip(true_labels, predicted_labels)
        if true_label == predicted_label
    )
    per_class = pd.DataFrame(
        {
            "class_name": labels,
            "family": [family_by_class[label] for label in labels],
            "training_images": [training_counts[label] for label in labels],
            "validation_images": support,
            "predicted_images": [predicted_counts[label] for label in labels],
            "true_positive": [true_positive[label] for label in labels],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    per_class.to_csv(output_path / "validation_metrics_by_class.csv", index=False)

    weighted = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    summary = {
        "model": str(model_path),
        "dataset": str(data_path),
        "classes": len(labels),
        "training_images": sum(training_counts.values()),
        "validation_images": len(true_labels),
        "top1_accuracy": float(np.mean(predictions["top1_correct"])),
        "top5_accuracy": float(np.mean(predictions["top5_correct"])),
        "precision": float(weighted[0]),
        "recall": float(weighted[1]),
        "f1": float(weighted[2]),
        "precision_recall_averaging": "support-weighted",
        "imgsz": imgsz,
    }
    (output_path / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(output_path / "validation_summary.csv", index=False)

    ordered = (
        per_class.sort_values(
            ["family", "training_images", "class_name"],
            ascending=[True, False, True],
        )["class_name"]
        .tolist()
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=ordered)
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )

    ordered_families = [family_by_class[label] for label in ordered]
    boundaries = [
        index
        for index in range(1, len(ordered_families))
        if ordered_families[index] != ordered_families[index - 1]
    ]

    fig, ax = plt.subplots(figsize=(28, 20))
    sns.heatmap(
        normalized,
        xticklabels=ordered,
        yticklabels=ordered,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        square=True,
        ax=ax,
        linewidths=0,
    )
    for position in boundaries:
        ax.axhline(position, color="darkblue", linewidth=0.5)
        ax.axvline(position, color="darkblue", linewidth=0.5)
    plt.title(
        "Normalized Confusion Matrix by Species (Grouped by Family, Ordered by Abundance)",
        fontsize=20,
    )
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix_grouped_colored.png", dpi=300)
    plt.savefig(output_path / "confusion_matrix_grouped_colored.pdf")
    plt.close()

    plot_data = per_class.set_index("class_name").loc[labels]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        plot_data["recall"],
        plot_data["precision"],
        c=plot_data["training_images"],
        cmap=plt.colormaps["PuRd"],
        s=100,
        alpha=0.9,
        edgecolors="black",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision vs Recall by Species\nColored by Number of Images", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Number of images", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "p_vs_r_species.png", dpi=300)
    plt.savefig(output_path / "p_vs_r_species.pdf")
    plt.close()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the 107-species classifier on validation images."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    summary = evaluate(
        args.model,
        args.data,
        args.taxonomy,
        args.output,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
