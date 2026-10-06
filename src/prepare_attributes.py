
from collections import defaultdict, Counter
from pathlib import Path
import json
from prepare_annotations import ROOT, csv_text, publish_files


def export_attributes(hasty_path, coco_path, output):
    hasty=json.loads(Path(hasty_path).read_text(encoding='utf8'))
    coco=json.loads(Path(coco_path).read_text(encoding='utf8'))
    images={i['id']:i for i in coco['images']}; cats={c['id']:c['name'] for c in coco['categories']}
    exact=defaultdict(list); candidates=defaultdict(list)
    for ann in coco['annotations']:
        key=(images[ann['image_id']]['file_name'],cats[ann['category_id']])
        candidates[key].append(ann['id'])
        seg=ann.get('segmentation')
        if isinstance(seg,list) and len(seg)==1: exact[(*key,tuple(seg[0]))].append(ann['id'])
    fields=[a['name'] for a in hasty['attributes']]
    rows=[]; unresolved=[]; counts={k:Counter() for k in fields}; statuses=Counter(); claimed=set()
    for im in hasty['images']:
        for lab in im['labels']:
            key=(im['image_name'],lab['class_name'])
            poly=tuple(v for xy in (lab.get('polygon') or []) for v in xy)
            matches=exact.get((*key,poly),[])
            matched=len(matches)==1 and matches[0] not in claimed
            if matched: claimed.add(matches[0])
            status='exact_geometry_match' if matched else 'needs_review'
            statuses[status]+=1
            row=dict(hasty_image_id=im['image_id'],hasty_label_id=lab['id'],file_name=im['image_name'],category_name=lab['class_name'],coco_annotation_id=matches[0] if matched else '',join_status=status)
            for k in fields:
                value=json.dumps(lab.get('attributes',{}).get(k),ensure_ascii=False)
                row[k]=value; counts[k][value]+=1
            rows.append(row)
            if not matched: unresolved.append(dict(hasty_label_id=lab['id'],file_name=im['image_name'],category_name=lab['class_name'],candidate_coco_annotation_ids=json.dumps(candidates.get(key,[]))))
    report={'hasty_images':len(hasty['images']),'hasty_instances':len(rows),'join_status_counts':dict(statuses),'attribute_value_counts':counts,'project_created':hasty.get('create_date'),'export_date':hasty.get('export_date'),'note':'null means not supplied; false and unknown are preserved separately. Project/export dates are not observation or annotation-period dates.'}
    files={'instance_attributes.csv':csv_text(rows,['hasty_image_id','hasty_label_id','file_name','category_name','coco_annotation_id','join_status']+fields),'join_needs_review.csv':csv_text(unresolved,['hasty_label_id','file_name','category_name','candidate_coco_annotation_ids']),'attribute_schema.json':json.dumps(hasty['attributes'],indent=2)+'\n','attribute_report.json':json.dumps(report,indent=2)+'\n'}
    publish_files(output,files)
    return {'instances':len(rows),**dict(statuses)}


if __name__=='__main__':
    print(export_attributes(ROOT/'data/additional_images_informations.json',ROOT/'data/final_data.json',ROOT/'data/prepared_attributes'))
