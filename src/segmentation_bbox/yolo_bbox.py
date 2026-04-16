import os
import json
from tqdm import tqdm

# =============================================================================
# yolo_bbox.py
# Converts COCO bounding-box annotations to YOLO format (.txt label files).
# Each output .txt file has the same base name as its corresponding image.
# Bounding-box coordinates are normalised to [0, 1] relative to image size.
#
# Input:
#   - data/final_data.json           : COCO annotation file
#
# Outputs:
#   - data/bbox/dataset/labels/      : one .txt per image (YOLO bbox format)
#   - data/bbox/dataset/classes.txt  : class ID -> name mapping
#
# Run:
#   python src/segmentation_bbox/yolo_bbox.py
# =============================================================================

# =============================================================================
# PATH CONFIGURATION — edit only this block
# =============================================================================
data_path       = os.path.join('data', 'final_data.json')
output_txt_dir  = os.path.join('data', 'bbox', 'dataset', 'labels')
class_file_path = os.path.join('data', 'bbox', 'dataset', 'classes.txt')
# =============================================================================


def load_coco_json(path):
    """Load a COCO JSON file and verify required keys are present."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for key in ['categories', 'images', 'annotations']:
        if key not in data:
            raise KeyError(f"Required key '{key}' missing in {path}")
    return data


def convert_bboxes_to_yolo(coco_data, output_dir, class_file):
    """
    Convert COCO bounding-box annotations to YOLO format and write .txt label files.

    COCO bbox format : [x_min, y_min, width, height]  (absolute pixels)
    YOLO bbox format : [class_id, x_center, y_center, width, height]  (normalised 0-1)

    Args:
        coco_data   (dict): Loaded COCO JSON data.
        output_dir  (str) : Directory where YOLO .txt files will be saved.
        class_file  (str) : Path to write the classes.txt file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build category index: COCO category_id -> sequential class index
    category_mapping = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    class_names      = [cat['name'] for cat in coco_data['categories']]

    with open(class_file, 'w', encoding='utf-8') as f:
        f.writelines(name + '
' for name in class_names)

    for img in tqdm(coco_data['images'], desc='Converting COCO bbox -> YOLO'):
        img_id       = img['id']
        img_width    = img['width']
        img_height   = img['height']
        img_stem     = os.path.splitext(img['file_name'])[0]

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        yolo_lines = []
        for ann in annotations:
            cat_id = ann['category_id']
            if cat_id not in category_mapping:
                continue  # skip unknown categories

            bbox     = ann['bbox']  # [x_min, y_min, w, h]
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width    = bbox[2] / img_width
            height   = bbox[3] / img_height

            yolo_lines.append(
                f'{category_mapping[cat_id]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}
'
            )

        txt_path = os.path.join(output_dir, f'{img_stem}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.writelines(yolo_lines)

    print(f'[OK] YOLO bbox labels written to: {output_dir}')
    print(f'[OK] classes.txt written to: {class_file}')


if __name__ == '__main__':
    data = load_coco_json(data_path)
    convert_bboxes_to_yolo(data, output_txt_dir, class_file_path)
