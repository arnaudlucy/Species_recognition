"""This script create a dataset split for our classification dataset with an 80/20 train/val ratio."""

import os
import shutil
import random
from math import ceil

# Set random seed for reproducibility
SEED = 40
random.seed(SEED)

# Paths
# source_dir: directory containing species subfolders (one subfolder per species)
# output_dir: directory where the dataset will be created, with train/ and val/ splits
source_dir = '/path/to/Photo_library_OB7'
output_dir = '/path/to/Yolo_classification/data_spp_seed'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# Create train/ and val/ directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through each species folder in the source directory
for species_folder in os.listdir(source_dir):
    species_path = os.path.join(source_dir, species_folder)

    # Skip if it's not a directory (e.g., ignore files at root level)
    if not os.path.isdir(species_path):
        continue

    # List all image files in the species folder
    images = [f for f in os.listdir(species_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Skip species with fewer than 2 images (cannot split properly)
    if len(images) < 2:
        continue

    # Shuffle the images to ensure randomness in train/val split
    random.shuffle(images)

    # Compute validation set size (20% of images, but at least 1 image if possible)
    val_count = max(1, int(0.2 * len(images)))
    val_images = images[:val_count]
    train_images = images[val_count:]

    # Create species-specific subfolders in train/ and val/
    species_train_dir = os.path.join(train_dir, species_folder)
    species_val_dir = os.path.join(val_dir, species_folder)
    os.makedirs(species_train_dir, exist_ok=True)
    os.makedirs(species_val_dir, exist_ok=True)

    # Copy training images into the corresponding train/ subfolder
    for img in train_images:
        shutil.copy2(os.path.join(species_path, img), os.path.join(species_train_dir, img))

    # Copy validation images into the corresponding val/ subfolder
    for img in val_images:
        shutil.copy2(os.path.join(species_path, img), os.path.join(species_val_dir, img))

print("Dataset successfully created with an 80/20 split per species (min. 1 image in validation).")
