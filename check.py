import os
import numpy as np
from collections import defaultdict

# === CONFIGURATION ===
ROOT = r"C:\Users\Lucy\Desktop\photoob7"
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# === FONCTIONS ===
def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXT


def count_images_per_folder(root):
    """Compte le nombre d'images dans chaque sous-dossier du dataset."""
    counts = defaultdict(int)
    ignored = []

    if not os.path.exists(root):
        print(f"⚠️ Folder not found: {root}")
        return counts, ignored

    for entry in sorted(os.listdir(root)):
        full_path = os.path.join(root, entry)
        if not os.path.isdir(full_path):
            continue

        n_images = sum(1 for f in os.listdir(full_path) if is_image(f))
        if n_images == 0:
            ignored.append(entry)
        else:
            counts[entry] = n_images

    return counts, ignored


def summarize_counts(counts):
    arr = np.array(list(counts.values()))
    if arr.size == 0:
        return {}
    return {
        "n_species": int(arr.size),
        "total_images": int(arr.sum()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "std": float(np.std(arr)),
        "n_gt_10": int((arr > 10).sum()),
    }


def display_summary(title, counts, ignored):
    summary = summarize_counts(counts)
    print(f"\n=== {title} ===")
    print("\n📊 Summary statistics")
    for k, v in summary.items():
        print(f"{k:<14}: {v}")

    if ignored:
        print("\n⚠️ Ignored empty folders:")
        for f in ignored:
            print(f"  {f}")

    print("\nTop 10 species by image count:")
    for sp, n in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {sp}: {n} images")

    return summary


# === MAIN ===
if __name__ == "__main__":
    print("📂 Checking Photo_library_OB7 dataset...\n")

    species_counts, ignored_folders = count_images_per_folder(ROOT)
    summary = display_summary("PHOTO_LIBRARY_OB7 (Global Dataset)", species_counts, ignored_folders)

    print(f"\n✅ Done. Found {summary['n_species']} folders with {summary['total_images']} images total.")
