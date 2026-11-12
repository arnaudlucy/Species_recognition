from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import LogLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
COCO_PATH = ROOT / "data" / "final_data.json"
OUTPUT_DIR = ROOT / "figures" / "data_description"


def load_category_counts():
    coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
    instance_counts = Counter(annotation["category_id"] for annotation in coco["annotations"])
    image_ids = defaultdict(set)
    for annotation in coco["annotations"]:
        image_ids[annotation["category_id"]].add(annotation["image_id"])

    rows = []
    for category in coco["categories"]:
        category_id = category["id"]
        rows.append(
            {
                "category_id": category_id,
                "category_name": category["name"],
                "populated": instance_counts[category_id] > 0,
                "annotated_instances": instance_counts[category_id],
                "unique_images": len(image_ids[category_id]),
            }
        )
    return coco, rows


def write_counts(rows):
    path = OUTPUT_DIR / "category_instance_counts.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_category_distribution(rows):
    populated = sorted(
        (row for row in rows if row["populated"]),
        key=lambda row: (-row["annotated_instances"], row["category_name"]),
    )
    ranks = list(range(1, len(populated) + 1))
    counts = [row["annotated_instances"] for row in populated]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    color = "#8d234a"
    ax.vlines(ranks, 1, counts, color="#d7afbf", linewidth=0.7, zorder=1)
    ax.scatter(ranks, counts, s=18, color=color, edgecolor="white", linewidth=0.35, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(0, len(populated) + 2)
    ax.set_ylim(0.9, max(counts) * 1.35)
    ax.set_xlabel("Populated taxonomic categories ranked by instance count")
    ax.set_ylabel("Annotated instances (log scale)")
    ax.set_title("Distribution of annotated instances across taxonomic categories", pad=12)
    ax.grid(axis="y", which="both", color="#e6e6e6", linewidth=0.7)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.spines[["top", "right"]].set_visible(False)

    leading = "\n".join(
        f"{row['category_name']} — {row['annotated_instances']}"
        for row in populated[:4]
    )
    ax.text(
        0.98,
        0.96,
        "Most represented categories\n" + leading,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.4"},
    )

    singleton_count = sum(row["annotated_instances"] == 1 for row in populated)
    ax.text(
        0.99,
        0.03,
        f"{len(populated)} populated categories; {singleton_count} with one instance\n"
        f"{len(rows) - len(populated)} declared categories have no annotations",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, "category_instance_distribution")


def rounded_box(ax, x, y, width, height, facecolor, title, detail, edgecolor=None):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor or facecolor,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x + width / 2, y + height * 0.29, detail, ha="center", va="center", fontsize=9)


def plot_selection_flow():
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dataset selection and annotation flow", pad=14)

    rounded_box(ax, 0.05, 0.58, 0.26, 0.23, "#dce9df", "2,564", "photographs selected")
    rounded_box(ax, 0.40, 0.58, 0.26, 0.23, "#dce4ef", "2,002", "annotated images in COCO")
    rounded_box(ax, 0.73, 0.58, 0.22, 0.23, "#ead9c4", "2,887", "annotated instances")

    arrow = dict(arrowstyle="-|>", color="#555555", linewidth=1.5, mutation_scale=14)
    ax.annotate("", xy=(0.40, 0.695), xytext=(0.31, 0.695), arrowprops=arrow)
    ax.annotate("", xy=(0.73, 0.695), xytext=(0.66, 0.695), arrowprops=arrow)
    ax.text(0.695, 0.73, "annotation", fontsize=8, color="#555555", ha="center")

    rounded_box(
        ax,
        0.05,
        0.19,
        0.26,
        0.19,
        "#f2f2f2",
        "562",
        "selected photographs not represented\nin the released COCO image table",
        edgecolor="#b8b8b8",
    )
    ax.annotate("", xy=(0.18, 0.38), xytext=(0.18, 0.58), arrowprops=arrow)
    ax.text(
        0.53,
        0.08,
        "The available records do not distinguish uncompleted annotations from other exclusions.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "image_selection_flow")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coco, rows = load_category_counts()
    if len(coco["images"]) != 2002 or len(coco["annotations"]) != 2887:
        raise ValueError("The COCO totals differ from the documented release")
    write_counts(rows)
    plot_category_distribution(rows)
    plot_selection_flow()


if __name__ == "__main__":
    main()
