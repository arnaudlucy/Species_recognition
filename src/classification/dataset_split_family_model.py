""" This script permit to create where species have been grouped by toxonoic families (e.g. Carcharhinidae) a dataset to train the family level model"""

import os
import shutil
import pandas as pd
import random
from math import ceil

# Set random seed for reproducibility
SEED = 4
random.seed(SEED)

# --- Data paths configuration ---
# Path to the directory containing test images
test_images_dir = "/data/photo_library"

# Path to the species dataset directory (contains train/val splits)
dataset_species_test_dir = "/data/yolo_classification/data_species_seed"

# Path to the CSV file containing species ecology information
species_ecology_csv = "/data/yolo_classification/species_ecology.csv"

# Output directory where images will be organized by family
output_dir = "/data/yolo_classification/dataset_family_seed"

# --- Load species ecology data ---
# Read the CSV file containing species and their corresponding families
df_ecology = pd.read_csv(species_ecology_csv)

# --- Create species to family mapping ---
# Dictionary to map species scientific names to their taxonomic families
species_to_family = {}

# Iterate through each row in the ecology dataframe
for idx, row in df_ecology.iterrows():
    # Normalize species name: replace underscores with spaces, strip whitespace, convert to lowercase
    species_name = row['species_scientific_name'].replace('_', ' ').strip().lower()
    family_name = row['Family']
    species_to_family[species_name] = family_name

# --- Build FAO code to species mapping ---
# Dictionary to map FAO codes to species names based on existing folder structure
code_to_species = {}

# Process both training and validation splits
for split in ["train", "val"]:
    split_path = os.path.join(dataset_species_test_dir, split)
    
    # Skip if split directory doesn't exist
    if not os.path.exists(split_path):
        continue
    
    # Process each folder in the split directory
    for folder_name in os.listdir(split_path):
        # Check if folder name contains FAO code and species name separated by hyphen
        if '-' in folder_name:
            # Split folder name to extract FAO code and species name
            fao_code, species_name = folder_name.split('-', 1)
            # Normalize species name: replace hyphens with spaces, strip whitespace, convert to lowercase
            species_name = species_name.replace('-', ' ').strip().lower()
            code_to_species[fao_code] = species_name

# --- Copy images to appropriate family directories ---
# Set to track species that don't have known family associations
unknown_species = set()
# Counter for successfully copied images
copied_images = 0

# Process each image file in the test images directory
for img_name in os.listdir(test_images_dir):
    # Skip macOS ghost files that start with "._"
    if img_name.startswith("._"):
        continue
    
    # Only process image files with common extensions
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    # Extract FAO code from image filename (code is before first underscore)
    fao_code = img_name.split('_')[0]
    
    # Check if FAO code exists in our mapping
    if fao_code in code_to_species:
        # Get species name from FAO code
        species_name = code_to_species[fao_code]
        # Get family name from species name
        family_name = species_to_family.get(species_name)
        
        # If family is known, copy image to appropriate family directory
        if family_name:
            # Create family directory if it doesn't exist
            family_folder = os.path.join(output_dir, family_name)
            os.makedirs(family_folder, exist_ok=True)
            
            # Copy image file to family directory
            src_path = os.path.join(test_images_dir, img_name)
            dst_path = os.path.join(family_folder, img_name)
            shutil.copy2(src_path, dst_path)
            copied_images += 1
        else:
            # Track species without known family
            unknown_species.add(species_name)
    else:
        # Warn about unrecognized FAO codes
        print(f"[WARNING] Unrecognized FAO code: {fao_code}")

# --- Summary and results ---
print("\nFinal Results:")
print(f"- {copied_images} images copied to {output_dir}")

# Display sample contents for debugging
print("Folders found in dataset_species_test_dir:", os.listdir(dataset_species_test_dir)[:10])
print("Images found in test_images_dir:", os.listdir(test_images_dir)[:10])
print("First few rows of CSV:", df_ecology.head())

# Report unknown species if any
if unknown_species:
    print("\nSpecies without known family association:")
    for species in sorted(unknown_species):
        print("-", species)
else:
    print("\All species are correctly associated with a family.")



""" In this second part we split the dataset into training and validation"""
import os
import shutil
import random
from math import ceil

# Set random seed for reproducibility
SEED = 5
random.seed(SEED)

# --- Data paths configuration ---
# Path to the input dataset directory containing family-organized images
input_dir = '/data/yolo_classification/dataset_family_seed'

# Base output directory where train/val splits will be created
output_base = '/data/yolo_classification/dataset_family_seed'

# --- Split ratios configuration ---
# Ratio of images to use for training (80%)
train_ratio = 0.8

# Ratio of images to use for validation (20% - calculated as 1 - train_ratio)
# Note: val_ratio = 0.28 in original code seems incorrect, should be 0.2
val_ratio = 0.2  # This equals 1 - train_ratio

# --- Create train and validation directories ---
# Create the main train and validation folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# --- Process each family directory ---
# Iterate through each family folder in the input directory
for family in os.listdir(input_dir):
    family_path = os.path.join(input_dir, family)
    
    # Skip if not a directory (ignore files in root)
    if not os.path.isdir(family_path):
        continue
    
    # --- Collect all image files for this family ---
    # Get all image files with common extensions
    images = [f for f in os.listdir(family_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Skip families with no images
    if not images:
        continue
    
    # Shuffle images randomly to ensure random distribution
    random.shuffle(images)
    
    # --- Calculate split indices ---
    # Calculate how many images go to training set
    split_idx = int(len(images) * train_ratio)
    
    # Split images into training and validation sets
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Family {family}: {len(train_images)} train, {len(val_images)} val images")
    
    # --- Move images to appropriate train/val directories ---
    # Process both training and validation splits
    for split, split_images in [('train', train_images), ('val', val_images)]:
        # Create family subfolder in train/val directory
        split_folder = os.path.join(output_base, split, family)
        os.makedirs(split_folder, exist_ok=True)
        
        # Move each image to the appropriate split directory
        for img in split_images:
            src_path = os.path.join(family_path, img)
            dst_path = os.path.join(split_folder, img)
            
            # Move file instead of copying to save disk space
            shutil.move(src_path, dst_path)
    
    # --- Clean up empty source directory ---
    # Remove the original family directory if it's now empty
    if not os.listdir(family_path):
        os.rmdir(family_path)
        print(f"Removed empty directory: {family_path}")

# --- Summary ---
print(f"\nDataset train/val split completed successfully!")
print(f"Output location: {output_base}")
print(f"Train ratio: {train_ratio} ({train_ratio*100}%)")
print(f"Validation ratio: {1-train_ratio} ({(1-train_ratio)*100}%)")

# Display final directory structure
print(f"\nFinal structure:")
for split in ['train', 'val']:
    split_path = os.path.join(output_base, split)
    if os.path.exists(split_path):
        families = os.listdir(split_path)
        print(f"- {split}: {len(families)} families")
        
        # Count total images in this split
        total_images = 0
        for family in families:
            family_path = os.path.join(split_path, family)
            if os.path.isdir(family_path):
                images = [f for f in os.listdir(family_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)
        print(f"  Total images: {total_images}")