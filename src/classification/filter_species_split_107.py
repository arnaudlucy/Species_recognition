from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile

from PIL import Image, ImageFile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXCLUDED_CLASSES = {
    "BRZ-Bramidae",
    "CAX-Ariidae",
    "FLF-Cantherhines-spp",
    "FLY-Exocoetidae",
}


def image_files(directory):
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith("._")
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_error(path):
    try:
        with Image.open(path) as image:
            image.load()
        return None
    except Exception as error:
        return str(error)


def repair_image(source, destination):
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(source) as image:
            image.load()
            image_format = image.format or "JPEG"
            exif = image.getexif()
            repaired = image.convert("RGB") if image_format.upper() in {"JPEG", "JPG"} else image.copy()
            options = {"format": image_format}
            if exif:
                options["exif"] = exif.tobytes()
            repaired.save(destination, **options)
        with Image.open(destination) as check:
            check.load()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_filtered_split(
    source,
    output,
    mode="hardlink",
    expected_class_count=107,
    excluded_classes=EXCLUDED_CLASSES,
):
    source = Path(source).resolve()
    output = Path(output).resolve()
    excluded_classes = set(excluded_classes)

    if mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError("mode must be symlink, hardlink or copy")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    split_directories = {split: source / split for split in ("train", "val")}
    if any(not path.is_dir() for path in split_directories.values()):
        raise FileNotFoundError("The source must contain train/ and val/ directories")

    original_classes = {
        split: {path.name for path in directory.iterdir() if path.is_dir()}
        for split, directory in split_directories.items()
    }
    missing_exclusions = excluded_classes - (
        original_classes["train"] | original_classes["val"]
    )
    if missing_exclusions:
        raise ValueError(
            "Excluded classes not found in the source: "
            + ", ".join(sorted(missing_exclusions))
        )

    retained_classes = {
        split: classes - excluded_classes
        for split, classes in original_classes.items()
    }
    if retained_classes["train"] != retained_classes["val"]:
        only_train = sorted(retained_classes["train"] - retained_classes["val"])
        only_val = sorted(retained_classes["val"] - retained_classes["train"])
        raise ValueError(
            f"Train/val class lists differ. Only train: {only_train}; only val: {only_val}"
        )
    if len(retained_classes["train"]) != expected_class_count:
        raise ValueError(
            f"Expected {expected_class_count} retained classes, found "
            f"{len(retained_classes['train'])}"
        )

    rows = []
    class_counts = []
    repaired_images = []
    hashes_by_split = defaultdict(set)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()

        for split in ("train", "val"):
            split_output = stage / split
            split_output.mkdir()
            for class_name in sorted(retained_classes[split]):
                class_source = split_directories[split] / class_name
                candidates = image_files(class_source)
                files = [(source_file, image_error(source_file)) for source_file in candidates]
                if not files:
                    raise ValueError(f"No images in {class_source}")
                if mode == "symlink" and any(error for _, error in files):
                    raise ValueError(
                        "Symlink mode cannot repair unreadable images; use hardlink or copy mode"
                    )

                class_output = split_output / class_name
                if mode == "symlink":
                    class_output.symlink_to(class_source, target_is_directory=True)
                else:
                    class_output.mkdir()

                class_counts.append(
                    {"split": split, "class_name": class_name, "images": len(files)}
                )
                for source_file, read_error in files:
                    relative_inside_class = source_file.relative_to(class_source)
                    destination = class_output / relative_inside_class
                    if mode != "symlink":
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if read_error:
                            repair_image(source_file, destination)
                            repaired_images.append(
                                {
                                    "split": split,
                                    "class_name": class_name,
                                    "relative_path": source_file.relative_to(source).as_posix(),
                                    "file_name": source_file.name,
                                    "source_error": read_error,
                                }
                            )
                        elif mode == "hardlink":
                            os.link(source_file, destination)
                        else:
                            shutil.copy2(source_file, destination)

                    prepared_file = destination if mode != "symlink" else source_file
                    source_digest = sha256(source_file)
                    digest = sha256(prepared_file)
                    hashes_by_split[digest].add(split)
                    rows.append(
                        {
                            "split": split,
                            "class_name": class_name,
                            "relative_path": source_file.relative_to(source).as_posix(),
                            "file_name": source_file.name,
                            "size_bytes": prepared_file.stat().st_size,
                            "source_sha256": source_digest,
                            "sha256": digest,
                            "preparation_action": "repaired_copy" if read_error else mode,
                        }
                    )

        duplicate_hashes = sorted(
            digest for digest, splits in hashes_by_split.items() if len(splits) > 1
        )
        if duplicate_hashes:
            raise ValueError(
                f"{len(duplicate_hashes)} byte-identical images cross train and val"
            )

        write_csv(
            stage / "split_manifest.csv",
            rows,
            [
                "split",
                "class_name",
                "relative_path",
                "file_name",
                "size_bytes",
                "source_sha256",
                "sha256",
                "preparation_action",
            ],
        )
        write_csv(
            stage / "class_counts.csv",
            class_counts,
            ["split", "class_name", "images"],
        )
        write_csv(
            stage / "repaired_images.csv",
            repaired_images,
            ["split", "class_name", "relative_path", "file_name", "source_error"],
        )
        report = {
            "source": str(source),
            "output": str(output),
            "mode": mode,
            "retained_classes": expected_class_count,
            "excluded_classes": sorted(excluded_classes),
            "images": dict(Counter(row["split"] for row in rows)),
            "repaired_unreadable_images": len(repaired_images),
            "duplicate_hashes_across_splits": duplicate_hashes,
            "ready_for_training": True,
        }
        (stage / "split_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(output)

    return report


def validate_filtered_split(
    dataset,
    expected_class_count=107,
    excluded_classes=EXCLUDED_CLASSES,
):
    dataset = Path(dataset).resolve()
    excluded_classes = set(excluded_classes)
    report_path = dataset / "split_report.json"
    manifest_path = dataset / "split_manifest.csv"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("The filtered dataset has no split report or manifest")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ready_for_training"):
        raise ValueError("The existing filtered dataset is not marked ready for training")

    classes = {}
    counts = {}
    for split in ("train", "val"):
        split_path = dataset / split
        if not split_path.is_dir():
            raise FileNotFoundError(split_path)
        class_paths = [path for path in split_path.iterdir() if path.is_dir()]
        symbolic_classes = [path.name for path in class_paths if path.is_symlink()]
        if symbolic_classes:
            raise ValueError(
                "Symbolic class directories are not supported by the YOLO classification "
                "dataset scanner. Rebuild the derived dataset in hardlink mode."
            )
        classes[split] = {path.name for path in class_paths}
        counts[split] = sum(len(image_files(path)) for path in class_paths)

    if classes["train"] != classes["val"]:
        raise ValueError("The existing train and val class lists differ")
    if len(classes["train"]) != expected_class_count:
        raise ValueError(
            f"Expected {expected_class_count} classes, found {len(classes['train'])}"
        )
    unexpected = classes["train"].intersection(excluded_classes)
    if unexpected:
        raise ValueError("Excluded classes remain: " + ", ".join(sorted(unexpected)))

    with manifest_path.open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    manifest_counts = Counter(row["split"] for row in manifest)
    if dict(manifest_counts) != counts:
        raise ValueError(
            f"Manifest counts {dict(manifest_counts)} do not match dataset counts {counts}"
        )

    return {
        "dataset": str(dataset),
        "classes": len(classes["train"]),
        "images": counts,
        "ready_for_training": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter an existing train/val classification split to 107 species classes."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-existing", type=Path)
    parser.add_argument(
        "--mode", choices=["symlink", "hardlink", "copy"], default="hardlink"
    )
    args = parser.parse_args()
    if args.validate_existing:
        report = validate_filtered_split(args.validate_existing)
    else:
        if args.source is None or args.output is None:
            parser.error("--source and --output are required when creating a split")
        report = prepare_filtered_split(args.source, args.output, mode=args.mode)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
