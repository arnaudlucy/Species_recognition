
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import io
import json
import math
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def csv_text(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def publish_files(destination, files):

    destination = Path(destination).resolve()
    if destination.exists():
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
        if actual != set(files) or any((destination/k).read_bytes() != v.encode('utf8') for k,v in files.items()):
            raise FileExistsError(f'{destination} contains different outputs. Choose a NEW output directory; source labels will not be overwritten.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=destination.parent) as tmp:
        stage = Path(tmp)/'output'
        stage.mkdir()
        for name, text in files.items():
            target = stage/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode('utf8'))
        stage.rename(destination)


def unique_index(items, kind):
    index = {}
    for item in items:
        key = item['id']
        if not isinstance(key, int) or isinstance(key, bool) or key in index:
            raise ValueError(f'Duplicate or non-integer {kind} ID: {key!r}')
        index[key] = item
    return index


def finite(values):
    return all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values)


def prepare(coco_path, output):

    raw = Path(coco_path).read_bytes()
    data = json.loads(raw)
    categories = unique_index(data['categories'], 'category')
    images = unique_index(data['images'], 'image')
    annotations = unique_index(data['annotations'], 'annotation')
    mapping = {cid:i for i,cid in enumerate(sorted(categories))}
    names = [categories[cid]['name'] for cid in sorted(categories)]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate category names')
    by_image = defaultdict(list)
    for ann in annotations.values():
        if ann['image_id'] not in images or ann['category_id'] not in categories:
            raise ValueError(f"Annotation {ann['id']}: unknown image or category ID")
        by_image[ann['image_id']].append(ann)
    files = {}; rows = []; manifest = []; stems = set(); counts = Counter()
    for image_id, im in images.items():
        width, height = im['width'], im['height']
        if not finite([width,height]) or width <= 0 or height <= 0:
            raise ValueError(f'Image {image_id}: invalid dimensions')
        name = im['file_name']
                                                                                                          
        if not isinstance(name,str) or '/' in name or '\\' in name or ':' in name or name in ('','.','..'):
            raise ValueError(f'Image {image_id}: expected a plain filename, got {name!r}')
        stem = PurePosixPath(name).stem
        if stem.casefold() in stems:
            raise ValueError(f'Label filename collision: {name}')
        stems.add(stem.casefold())
        seg_lines=[]; box_lines=[]
        for ann in by_image[image_id]:
            aid=ann['id']; cid=ann['category_id']; bbox=ann.get('bbox')
            if not isinstance(bbox,list) or len(bbox)!=4 or not finite(bbox):
                raise ValueError(f'Annotation {aid}: invalid bbox')
            x,y,w,h=bbox
            if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height:
                raise ValueError(f'Annotation {aid}: bbox outside image or zero area')
            seg=ann.get('segmentation')
            if not isinstance(seg,list) or len(seg)!=1:
                raise ValueError(f'Annotation {aid}: expected one polygon per instance. Multipart/RLE/empty segmentation needs an explicit conversion policy; no labels were written.')
            poly=seg[0]
            if not isinstance(poly,list) or len(poly)<6 or len(poly)%2 or not finite(poly):
                raise ValueError(f'Annotation {aid}: polygon needs at least three finite x/y pairs')
            if any(v<0 or v>(width if i%2==0 else height) for i,v in enumerate(poly)):
                raise ValueError(f'Annotation {aid}: polygon outside image')
            if len(set(zip(poly[::2],poly[1::2])))<3:
                raise ValueError(f'Annotation {aid}: fewer than three distinct polygon points')
            coords=[v/(width if i%2==0 else height) for i,v in enumerate(poly)]
            yid=mapping[cid]
            seg_lines.append(str(yid)+' '+ ' '.join(f'{v:.6f}' for v in coords)+'\n')
            box_lines.append(str(yid)+' '+ ' '.join(f'{v:.6f}' for v in ((x+w/2)/width,(y+h/2)/height,w/width,h/height))+'\n')
            rows.append(dict(annotation_id=aid,image_id=image_id,file_name=name,source_category_id=cid,species_id=yid,species_name=categories[cid]['name'],label_line=len(seg_lines)))
            counts[cid]+=1
        files[f'segmentation/labels/{stem}.txt']=''.join(seg_lines)
        files[f'bbox/labels/{stem}.txt']=''.join(box_lines)
        manifest.append(dict(image_id=image_id,file_name=name,width=width,height=height,instances=len(seg_lines)))
    class_rows=[dict(source_category_id=cid,yolo_id=mapping[cid],category_name=categories[cid]['name'],instances=counts[cid]) for cid in sorted(categories)]
    files['classes.txt']='\n'.join(names)+'\n'                                          
    files['class_map.csv']=csv_text(class_rows,['source_category_id','yolo_id','category_name','instances'])
    files['annotations.csv']=csv_text(rows,['annotation_id','image_id','file_name','source_category_id','species_id','species_name','label_line'])
    files['image_manifest.csv']=csv_text(manifest,['image_id','file_name','width','height','instances'])
    report=dict(images=len(images),annotations=len(annotations),declared_categories=len(categories),represented_categories=len(counts),source_sha256=hashlib.sha256(raw).hexdigest(),multipart_policy='reject; never split one instance into multiple YOLO instances',class_order='ascending original COCO category ID')
    files['validation.json']=json.dumps(report,indent=2)+'\n'
    publish_files(output,files)
    return report


def normalize(name):
    return ' '.join(name.replace('_',' ').replace('-',' ').casefold().split())


def family_mapping(class_names, ecology_csv, asfis_csv):

    with Path(ecology_csv).open(encoding='utf-8-sig',newline='') as f:
        ecology={normalize(r['species_scientific_name']):r['Family'].strip() for r in csv.DictReader(f)}
    with Path(asfis_csv).open(encoding='utf-8-sig',newline='') as f:
        asfis={normalize(r['Scientific_Name']):r['Family'].strip().capitalize() for r in csv.DictReader(f)}
                                                                                                                    
    overrides={'unknown shark':'Carcharhinidae','unknown turtle':'Cheloniidae','unknown bonyfish':'Unknown_bonyfish'}
    aliases={'unknown cantherhines':'unknown cantherines','unknown brama':'unknown bramidae'}
    rows=[]
    for cid,name in enumerate(class_names):
        taxon=name.split('-',1)[1] if '-' in name else name
        key=normalize(taxon); lookup=aliases.get(key,key)
        if key in overrides:
            family=overrides[key]; source='author-confirmed collection identification' if key!='unknown bonyfish' else 'unresolved group (not a biological family)'
        else:
            family=ecology.get(lookup); source='species_ecology.csv'+(' (explicit spelling/group alias)' if lookup!=key else '')
            if family in (None,'','N/A'):
                family=asfis.get(lookup); source='ASFIS_sp_2025.csv, exact scientific-name fallback'
        if not family or family=='N/A':
            raise ValueError(f'No family mapping for {name!r}; no objects will be silently removed')
        rows.append(dict(species_id=cid,category_name=name,family=family,mapping_source=source))
    families=sorted(set(r['family'] for r in rows))
    for row in rows: row['family_id']=families.index(row['family'])
    return rows,families


def derive_families(prepared, output, ecology_csv=ROOT/'notebooks/species_ecology.csv', asfis_csv=ROOT/'notebooks/ASFIS_sp_2025.csv'):
    prepared=Path(prepared)
    names=(prepared/'classes.txt').read_text(encoding='utf8').splitlines()
    mapping,families=family_mapping(names,ecology_csv,asfis_csv)
    files={'classes.txt':'\n'.join(families)+'\n','family_map.csv':csv_text(mapping,['species_id','category_name','family','mapping_source','family_id'])}
    total={}
    for task in ('segmentation','bbox'):
        total[task]=0
        for path in sorted((prepared/task/'labels').glob('*.txt')):
            lines=[]
            for line in path.read_text(encoding='utf8').splitlines():
                parts=line.split()
                if not parts: continue
                sid=int(parts[0])
                if sid<0 or sid>=len(mapping): raise ValueError(f'{path}: unknown ID {sid}')
                parts[0]=str(mapping[sid]['family_id']); lines.append(' '.join(parts)+'\n'); total[task]+=1
            files[f'{task}/labels/{path.name}']=''.join(lines)
    files['validation.json']=json.dumps({'instances_by_task':total,'family_output_classes':len(families),'includes_unresolved_bonyfish_group':True,'note':'Training output classes are not the manuscript biological-family count.'},indent=2)+'\n'
    publish_files(output,files)
    return total


def derive_rare(prepared, output, threshold=10):

    if not isinstance(threshold,int) or threshold<1: raise ValueError('Threshold must be a positive integer')
    prepared=Path(prepared)
    if (prepared/'rare_mapping.json').exists():
        raise ValueError('Input has already been merged; use the original prepared annotations')
    names=(prepared/'classes.txt').read_text(encoding='utf8').splitlines()
    with (prepared/'annotations.csv').open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
    counts=Counter(int(r['species_id']) for r in rows)
    kept=[sid for sid in range(len(names)) if counts[sid]>=threshold]
    rare=[sid for sid in range(len(names)) if 0<counts[sid]<threshold]
    mapping={sid:i for i,sid in enumerate(kept)}
    new_names=[names[sid] for sid in kept]
    if rare:
        mapping.update({sid:len(new_names) for sid in rare}); new_names.append('rare_species')
    files={}
    for task in ('bbox','segmentation'):
        for path in sorted((prepared/task/'labels').glob('*.txt')):
            lines=[]
            for line in path.read_text(encoding='utf8').splitlines():
                tokens=line.split()
                if not tokens: continue
                sid=int(tokens[0])
                if sid not in mapping: raise ValueError(f'{path}: unmapped class {sid}')
                tokens[0]=str(mapping[sid]); lines.append(' '.join(tokens)+'\n')
            files[f'{task}/labels/{path.name}']=''.join(lines)
    for row in rows:
        row['species_id']=mapping[int(row['species_id'])]
        row['species_name']=new_names[row['species_id']]
    files['annotations.csv']=csv_text(rows,['annotation_id','image_id','file_name','source_category_id','species_id','species_name','label_line'])
    files['image_manifest.csv']=(prepared/'image_manifest.csv').read_text(encoding='utf8')
    files['classes.txt']='\n'.join(new_names)+'\n'
    files['rare_mapping.json']=json.dumps({'threshold':threshold,'old_to_new':mapping,'source_classes':names,'note':'Optional policy based on full annotation counts, as in the historical notebook. Apply BEFORE making a new split; not used to reinterpret old checkpoints.'},indent=2)+'\n'
    publish_files(output,files)
    return {'annotations':len(rows),'output_classes':len(new_names),'merged_source_classes':len(rare)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco',type=Path,default=ROOT/'data/final_data.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/prepared')
    parser.add_argument('--families',action='store_true')
    args=parser.parse_args()
    print(json.dumps(prepare(args.coco,args.output),indent=2))
    if args.families: print(derive_families(args.output,args.output.with_name(args.output.name+'_families')))


if __name__=='__main__':
    main()
