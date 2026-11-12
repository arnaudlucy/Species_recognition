from pathlib import Path
import argparse
import hashlib
import json
import shutil

from ultralytics import YOLO

from evaluate_species_classification import evaluate


def train(args):
    project = Path(args.project).resolve()
    run_directory = project / args.name
    if run_directory.exists():
        raise FileExistsError(f"Training run already exists: {run_directory}")

    model = YOLO(args.model, task="classify")
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        seed=0,
        deterministic=True,
        dropout=0.2,
        cos_lr=False,
        lr0=0.0001,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.01,
        warmup_epochs=3.0,
        translate=0.5,
        scale=0.5,
        fliplr=0.5,
        mixup=0.0,
        auto_augment="randaugment",
        erasing=0.2,
        val=True,
        plots=True,
    )

    best_model = run_directory / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(best_model)

    evaluation_directory = run_directory / "publication_evaluation"
    summary = evaluate(
        best_model,
        args.data,
        args.taxonomy,
        evaluation_directory,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    dataset = Path(args.data).resolve()
    for name in (
        "split_manifest.csv",
        "class_counts.csv",
        "repaired_images.csv",
        "split_report.json",
    ):
        source = dataset / name
        if source.is_file():
            shutil.copy2(source, evaluation_directory / name)

    digest = hashlib.sha256(best_model.read_bytes()).hexdigest()
    (evaluation_directory / "best.pt.sha256").write_text(
        f"{digest}  best.pt\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the 107-species YOLO classifier."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--model", default="yolo11l-cls.pt")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
