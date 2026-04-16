import os
import pandas as pd
from tqdm import tqdm

# =============================================================================
# bbox_to_family.py
# Converts YOLO bounding-box label files from species-level to family-level.
# Reads the species->family mapping from species_ecology.csv, then rewrites
# every .txt label file replacing species class IDs with family class IDs.
#
# Inputs  (edit the PATH CONFIGURATION block below):
#   - species_ecology.csv            : taxonomic mapping table
#   - data/bbox/dataset/labels/      : YOLO .txt bbox label files (species-level)
#   - data/bbox/classes.txt          : ordered species class list (one name per line)
#
# Outputs:
#   - data/bbox/dataset/labels_by_family/            : rewritten label files (family-level)
#   - data/bbox/dataset/labels_by_family/classes.txt : family class names
#   - data/bbox/dataset/labels_by_family/family_id_map.txt : ID -> family name mapping
#
# Run:
#   python src/segmentation_bbox/bbox_to_family.py
# =============================================================================

# =============================================================================
# PATH CONFIGURATION — edit only this block
# =============================================================================
csv_path            = os.path.join('notebooks', 'species_ecology.csv')
input_labels_dir    = os.path.join('data', 'bbox', 'dataset', 'labels')
original_classes    = os.path.join('data', 'bbox', 'classes.txt')
output_labels_dir   = os.path.join('data', 'bbox', 'dataset', 'labels_by_family')
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
    'Unknown_bonyfish':     'Unknown_bonyfish',
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
family_to_id = {fam: i for i, fam in enumerate(unique_families)}

os.makedirs(output_labels_dir, exist_ok=True)

# Rewrite label files replacing species IDs with family IDs
for filename in tqdm(os.listdir(input_labels_dir), desc='Converting bbox labels to family level'):
    if not filename.endswith('.txt'):
        continue

    input_path = os.path.join(input_labels_dir, filename)
    output_path = os.path.join(output_labels_dir, filename)

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if not parts[0].isdigit():
                        print(f'[WARNING] Skipping {filename}: invalid class token -> {parts[0]}')
                        break
                    old_class_id = int(parts[0])
                    if old_class_id not in species_idx_to_family:
                        continue  # no family for this class; skip
                    new_class_id = family_to_id[species_idx_to_family[old_class_id]]
                    outfile.write('{} {}\n'.format(new_class_id, ' '.join(parts[1:])))
    except Exception as e:
        print('[ERROR] {}: {}'.format(filename, e))

# Write family classes.txt
with open(output_classes_path, 'w', encoding='utf-8') as f:
    for fam in unique_families:
        f.write(fam + '\n')

# Write family_id_map.txt
with open(output_family_map, 'w', encoding='utf-8') as f:
    for fam, fam_id in family_to_id.items():
        f.write('{} {}\n'.format(fam_id, fam))

# Summary
if unknown_species:
    print('[WARNING] {} species without a known family:'.format(len(unknown_species)))
    for s in unknown_species:
        print('  - {}'.format(s))
else:
    print('[OK] All species have a known family.')

print('[OK] Family-level bbox labels written to: {}'.format(output_labels_dir))
print('[OK] classes.txt and family_id_map.txt generated.')
