import os
import json
from unicodedata import category
import cv2
import numpy as np

from collections import defaultdict
from detectron2.structures import BoxMode

# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def get_vg_dicts(root, split):
    json_file = os.path.join(root, 'annotations', f"visual_genome_{split}.json")
    image_dir1 = os.path.join(root, 'VG_100K')
    image_dir2 = os.path.join(root, 'VG_100K_2')
    image_set1 = set(os.listdir(image_dir1))
    image_set2 = set(os.listdir(image_dir2))
    with open(json_file) as f:
        data_dict = json.load(f)
    image_list = data_dict['images']
    anno_list = data_dict['annotations']
    
    imgid2meta = {}
    for item in image_list:
        imgid2meta[item['id']] = item
    imgid2anno = defaultdict(list)
    for item in anno_list:
        imgid2anno[item['image_id']].append(item)
    # catid2name = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}

    dataset_dicts = []
    for imgid, meta in imgid2meta.items():
        record = {}
        if meta['file_name'] in image_set1:
            record['file_name'] = os.path.join(image_dir1, meta['file_name'])
        elif meta['file_name'] in image_set2:
            record['file_name'] = os.path.join(image_dir2, meta['file_name'])
        else:
            print(f'Image {record["file_name"]} not found')
            raise FileNotFoundError
        record['image_id'] = imgid
        record['height'] = meta['height']
        record['width'] = meta['width']
        annos = imgid2anno[imgid]
        for i in range(len(annos)):
            annos[i]['bbox_mode'] = BoxMode.XYXY_ABS
            x, y, h, w = annos[i]['bbox']
            x1, y1, x2, y2 = x, y, x+h, y+w
            annos[i]['bbox'] = (x1, y1, x2, y2)

        record['annotations'] = annos
        dataset_dicts.append(record)
    return dataset_dicts

with open('data/VisualGenome/annotations/visual_genome_test.json', 'r') as f:
    categories = json.load(f)['categories']
categories = [c['name'] for c in categories]

for d in ['train', 'val', 'test']:
    DatasetCatalog.register('vg_'+d, lambda d=d: get_vg_dicts('data/VisualGenome/', split=d))
    MetadataCatalog.get('vg_'+d).set(thing_classes=categories)
    MetadataCatalog.get('vg_'+d).set(evaluator_type='coco')
del categories

vg_metadata = MetadataCatalog.get('vg_train')

if __name__ == '__main__':
    dataset_dicts = get_vg_dicts('data/VisualGenome', 'train')
    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:,:,::-1], metadata=vg_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        temp = d
        cv2.imwrite(f'{d["image_id"]}.jpg', out.get_image()[:,:,::-1])

