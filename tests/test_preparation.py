import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_annotations import prepare, derive_families, derive_rare
from prepare_splits import classification_split, copy_segmentation_split


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.sample={'categories':[{'id':9,'name':'RSK-Unknown-shark'},{'id':4,'name':'BET-Thunnus-obesus'}],'images':[{'id':2,'file_name':'fish.jpg','width':100,'height':50}],'annotations':[{'id':8,'image_id':2,'category_id':9,'bbox':[10,5,20,10],'segmentation':[[10,5,30,5,30,15,10,15]]}]}
    def tearDown(self): self.tmp.cleanup()
    def run_sample(self,data=None):
        p=self.root/'input.json'; p.write_text(json.dumps(data or self.sample),encoding='utf8')
        return prepare(p,self.root/'output')
    def test_sparse_ids_and_coordinates(self):
        self.run_sample()
        self.assertEqual((self.root/'output/bbox/labels/fish.txt').read_text(),'1 0.200000 0.200000 0.200000 0.200000\n')
        self.assertEqual((self.root/'output/classes.txt').read_text().splitlines(),['BET-Thunnus-obesus','RSK-Unknown-shark'])
        self.assertEqual((self.root/'output/segmentation/labels/fish.txt').read_text().split()[0],'1')
    def test_multipart_fails_before_writing(self):
        self.sample['annotations'][0]['segmentation'].append([40,20,50,20,50,30])
        with self.assertRaisesRegex(ValueError,'Multipart'): self.run_sample()
        self.assertFalse((self.root/'output').exists())
    def test_malformed_polygon_diagnostic(self):
        self.sample['annotations'][0]['segmentation'][0].append(1)
        with self.assertRaisesRegex(ValueError,'Annotation 8'): self.run_sample()
    def test_unknown_category_not_silently_skipped(self):
        self.sample['annotations'][0]['category_id']=999
        with self.assertRaisesRegex(ValueError,'unknown image or category'): self.run_sample()
    def test_path_traversal_rejected(self):
        self.sample['images'][0]['file_name']='../fish.jpg'
        with self.assertRaisesRegex(ValueError,'plain filename'): self.run_sample()
    def test_empty_image_gets_empty_label(self):
        self.sample['annotations']=[]; self.run_sample()
        self.assertEqual((self.root/'output/segmentation/labels/fish.txt').read_text(),'')
    def test_identical_rerun_and_conflicting_output(self):
        self.run_sample(); self.run_sample()
        (self.root/'output/classes.txt').write_text('changed')
        with self.assertRaises(FileExistsError): self.run_sample()
    def test_family_keeps_objects_and_author_identification(self):
        self.run_sample(); original=(self.root/'output/segmentation/labels/fish.txt').read_bytes()
        counts=derive_families(self.root/'output',self.root/'family')
        self.assertEqual(counts,{'bbox':1,'segmentation':1})
        names=(self.root/'family/classes.txt').read_text().splitlines()
        token=int((self.root/'family/segmentation/labels/fish.txt').read_text().split()[0])
        self.assertEqual(names[token],'Carcharhinidae')
        self.assertEqual((self.root/'output/segmentation/labels/fish.txt').read_bytes(),original)
    def test_rare_merge_does_not_mutate_or_merge_twice(self):
        self.run_sample(); original=(self.root/'output/segmentation/labels/fish.txt').read_bytes()
        self.assertEqual(derive_rare(self.root/'output',self.root/'rare')['annotations'],1)
        self.assertEqual((self.root/'rare/classes.txt').read_text(),'rare_species\n')
        self.assertEqual((self.root/'output/segmentation/labels/fish.txt').read_bytes(),original)
        with self.assertRaisesRegex(ValueError,'already been merged'):
            derive_rare(self.root/'rare',self.root/'twice')
    def test_classification_rare_policy_and_determinism(self):
        source=self.root/'photos'
        for cls,n in [('singleton',1),('pair',2),('common',10)]:
            (source/cls).mkdir(parents=True)
            for i in range(n): (source/cls/f'{i}.jpg').write_bytes(f'{cls}-{i}'.encode())
        r=classification_split(source,self.root/'split')
        classification_split(source,self.root/'split2')
        self.assertEqual((self.root/'split/split_manifest.csv').read_bytes(),(self.root/'split2/split_manifest.csv').read_bytes())
        self.assertEqual(r['counts'],{'val':3,'train':9})
        self.assertEqual(r['excluded_classes'][0]['category'],'singleton')
        self.assertEqual(len(list(source.rglob('*.jpg'))),13)
    def test_missing_images_do_not_create_partial_dataset(self):
        self.run_sample(); photos=self.root/'photos'; photos.mkdir()
        manifest=self.root/'split.csv'; manifest.write_text('file_name,split\nfish.jpg,train\n')
        with self.assertRaisesRegex(ValueError,'Dataset was not copied'):
            copy_segmentation_split(self.root/'output',manifest,photos,self.root/'dataset')
        self.assertFalse((self.root/'dataset').exists())
    def test_duplicate_classification_images_flagged(self):
        source=self.root/'photos/a'; source.mkdir(parents=True)
        for i in range(2): (source/f'{i}.jpg').write_bytes(b'same-image')
        report=classification_split(source.parent,self.root/'split')
        self.assertFalse(report['ready_for_evaluation'])


if __name__=='__main__': unittest.main()
