import os
import pandas as pd
from tqdm import tqdm

# =============================================================================
# seg_to_family.py
# Converts YOLO segmentation label files from species-level to family-level.
# Identical logic to bbox_to_family.py but operates on polygon label files.
#
# Inputs  (edit the PATH CONFIGURATION block below):
#   - species_ecology.csv       : taxonomic mapping table
#   - labels/                   : YOLO .txt segmentation label files (species-level)
#   - classes.txt               : ordered species class list (one name per line)
#
# Outputs:
#   - labels_by_family/         : rewritten .txt label files (family-level)
#   - labels_by_family/classes.txt      : family class names
#   - labels_by_family/family_id_map.txt: family ID -> name mapping
# =============================================================================

# =============================================================================
# PATH CONFIGURATION — edit only this block
# =============================================================================
csv_path            = os.path.join('notebooks', 'species_ecology.csv')
input_labels_dir    = os.path.join('data', 'segmentation', 'labels')
original_classes    = os.path.join('data', 'bbox', 'classes.txt')   # shared with bbox
output_labels_dir   = os.path.join('data', 'segmentation', 'labels_by_family')
output_classes_path = os.path.join(output_labels_dir, 'classes.txt')
output_family_map   = os.path.join(output_labels_dir, 'family_id_map.txt')
# =============================================================================

# Load species -> family mapping from CSV
df = pd.read_csv(csv_path)
species_to_family = dict(zip(df['species_scientific_name'], df['Family']))

# Manual overrides for unknown/group labels not present in the CSV
manual_families = {
    'Unknown_shark':        'Carcharhinidae',
    'Unknown_turtle':       'Cheloniidae',
    'Unknown_scombridae':   'Scombridae',
    'Unknown_bonyfish':     'unknown_bonyfish',
    'Unknown_cantherhines': 'Monacanthidae',
    'unknown_carangidae':   'Carangidae',
    'Unknown_brama':        'Bramidae',
}
species_to_family.update(manual_families)

# Load ordered species list from classes.txt
with open(original_classes, 'r', encoding='utf-8') as f:
    species_list = [line.strip().replace("'", '') for line in f]

# Map each species index to its family
species_idx_to_family = {}
unknown_species = []
for idx, species in enumerate(species_list):
    family = species_to_family.get(species)
    if not family or (isinstance(family, float) and pd.isna(family)):
        unknown_species.append(species)
    else:
        species_idx_to_family[idx] = family

# Build family -> ID mapping (sorted for reproducibility)
unique_families = sorted(set(species_idx_to_family.values()))
family_to_id    = {fam: i for i, fam in enumerate(unique_families)}

os.makedirs(output_labels_dir, exist_ok=True)

# Rewrite segmentation label files
for filename in tqdm(os.listdir(input_labels_dir), desc='Converting segmentation labels to family level'):
    if not filename.endswith('.txt'):
        continue
    input_path  = os.path.join(input_labels_dir, filename)
    output_path = os.path.join(output_labels_dir, filename)
    try:
        with open(input_path, 'r', encoding='utf-8') as infile,              open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                parts = line.strip().split()
                if not parts:
                    continue
                if not parts[0].isdigit():
                    print(f'[WARNING] Skipping {filename}: invalid class token -> {parts[0]}')
                    break
                old_class_id = int(parts[0])
                if old_class_id not in species_idx_to_family:
                    continue
                new_class_id = family_to_id[species_idx_to_family[old_class_id]]
                outfile.write(' '.join([str(new_class_id)] + parts[1:]) + '
')
    except Exception as e:
        print(f'[ERROR] {filename}: {e}')

# Write output files
with open(output_classes_path, 'w', encoding='utf-8') as f:
    for fam in unique_families:
        f.write(f'{fam}
')

with open(output_family_map, 'w', encoding='utf-8') as f:
    for fam, fam_id in family_to_id.items():
        f.write(f'{fam_id} {fam}
')

# Summary
if unknown_species:
    print(f'[WARNING] {len(unknown_species)} species without a known family:')
    for s in unknown_species:
        print(f'  - {s}')
else:
    print('[OK] All species have a known family.')

print('[OK] Family-level segmentation labels conversion complete.')
print(f'[OK] Labels written to: {output_labels_dir}')
print('[OK] classes.txt and family_id_map.txt generated.')
