
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from prepare_annotations import csv_text, publish_files

IMAGE_EXTENSIONS={'.jpg','.jpeg','.png'}


def segmentation_split(prepared, output, fraction=0.2, seed=42, cap=0.2):

    if not 0<fraction<1 or not 0<cap<1: raise ValueError('Ratios must lie between 0 and 1')
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    prepared=Path(prepared)
    image_to_species=defaultdict(set)
    with (prepared/'image_manifest.csv').open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f): image_to_species[row['file_name']]=set()
    with (prepared/'annotations.csv').open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f): image_to_species[row['file_name']].add(int(row['species_id']))
    files=sorted(image_to_species)
    if len(files)<2: raise ValueError('At least two images are required for a split')
    species=sorted(set().union(*image_to_species.values()))
    y=np.array([[int(s in image_to_species[f]) for s in species] for f in files])
    splitter=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=fraction,random_state=seed)
    ti,vi=next(splitter.split(np.arange(len(files)).reshape(-1,1),y))
    train=[files[i] for i in ti]; val=[files[i] for i in vi]
    initial={'train':len(train),'val':len(val)}
    totals=Counter(s for f in files for s in image_to_species[f])
                                                                                          
    while True:
        vc=Counter(s for f in val for s in image_to_species[f])
        over=next((s for s in species if vc[s]/totals[s]>cap),None)
        if over is None: break
        moved=next(f for f in val if over in image_to_species[f])
        val.remove(moved); train.append(moved)
    members=[dict(file_name=f,split=split) for split,group in [('train',train),('val',val)] for f in sorted(group)]
    vc=Counter(s for f in val for s in image_to_species[f])
    coverage=[dict(species_id=s,total_images=totals[s],train_images=totals[s]-vc[s],val_images=vc[s]) for s in species]
    report=dict(seed=seed,target_validation_fraction=fraction,per_label_validation_cap=cap,initial=initial,final={'train':len(train),'val':len(val)},actual_validation_fraction=len(val)/len(files),classes_without_validation=[s for s in species if vc[s]==0],note='New preparation manifest, not proof of membership in historical model runs. Image availability/duplicate checks happen before copying.')
    publish_files(output,{'split_manifest.csv':csv_text(members,['file_name','split']),'class_coverage.csv':csv_text(coverage,['species_id','total_images','train_images','val_images']),'split_report.json':json.dumps(report,indent=2)+'\n'})
    return report


def classification_split(source, output, seed=40, fraction=0.2):

    source=Path(source).resolve()
    if not source.is_dir(): raise FileNotFoundError(source)
    if not 0<fraction<1: raise ValueError('Ratio must lie between 0 and 1')
    rng=random.Random(seed); rows=[]; skipped=[]; hash_splits=defaultdict(set)
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        images=sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith('._'))
        if len(images)<2:
            skipped.append(dict(category=directory.name,images=len(images),reason='original policy: fewer than two images'))
            continue
        rng.shuffle(images)
        n=max(1,int(fraction*len(images)))
        for i,path in enumerate(images):
            split='val' if i<n else 'train'
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            hash_splits[digest].add(split)
            rows.append(dict(relative_path=path.relative_to(source).as_posix(),category=directory.name,split=split,sha256=digest))
    overlaps=[h for h,s in hash_splits.items() if len(s)>1]
    report={'seed':seed,'target_validation_fraction':fraction,'counts':dict(Counter(r['split'] for r in rows)),'excluded_classes':skipped,'duplicate_hashes_across_splits':overlaps,'ready_for_evaluation':not overlaps,'note':'A duplicate crossing splits requires reconciliation before use; no historical split is overwritten.'}
    publish_files(output,{'split_manifest.csv':csv_text(rows,['relative_path','category','split','sha256']),'split_report.json':json.dumps(report,indent=2)+'\n'})
    return report


def copy_segmentation_split(prepared, manifest, image_root, output, task='segmentation'):

    if task not in ('segmentation','bbox'): raise ValueError('Unknown task')
    prepared=Path(prepared); image_root=Path(image_root).resolve(); output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Choose a new dataset output directory')
    if not image_root.is_dir(): raise FileNotFoundError(image_root)
    by_name=defaultdict(list)
    for p in image_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith('._'): by_name[p.name].append(p)
    with Path(manifest).open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
    with (prepared/'image_manifest.csv').open(encoding='utf8',newline='') as f: expected={r['file_name'] for r in csv.DictReader(f)}
    if len(rows)!=len(expected) or {r['file_name'] for r in rows}!=expected:
        raise ValueError('Split manifest must contain each prepared image exactly once')
    errors=[]; plan=[]; hashes=defaultdict(set)
    for row in rows:
        name=row['file_name']; split=row['split']; hits=by_name.get(name,[])
        label=prepared/task/'labels'/f'{Path(name).stem}.txt'
        if split not in ('train','val') or len(hits)!=1 or not label.is_file():
            errors.append(f'{name}: images={len(hits)}, label={label.is_file()}, split={split}'); continue
        digest=hashlib.sha256(hits[0].read_bytes()).hexdigest(); hashes[digest].add(split)
        plan.append((row,hits[0],label))
    if any(len(s)>1 for s in hashes.values()): errors.append('Byte-identical photographs cross train/validation splits')
    if errors: raise ValueError('Dataset was not copied. Fix source availability/manifest first:\n'+'\n'.join(errors))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        stage=Path(td)/'dataset'; stage.mkdir()
        for row,im,lab in plan:
            for kind,src in [('images',im),('labels',lab)]:
                dst=stage/kind/row['split']/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        shutil.copy2(prepared/'classes.txt',stage/'classes.txt')
        shutil.copy2(manifest,stage/'split_manifest.csv')
        stage.rename(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['segmentation','classification'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fn=segmentation_split if args.task=='segmentation' else classification_split
    print(json.dumps(fn(args.source,args.output),indent=2))
