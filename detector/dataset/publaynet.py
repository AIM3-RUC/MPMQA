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


def get_publaynet_dicts(root, split):
    json_file = os.path.join(root, f"{split}.json")
    image_dir = os.path.join(root, split)
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
        record['file_name'] = os.path.join(image_dir, meta['file_name'])
        record['image_id'] = imgid
        record['height'] = meta['height']
        record['width'] = meta['width']
        annos = imgid2anno[imgid]
        for i in range(len(annos)):
            poly = annos[i]['segmentation'][0]
            px = poly[::2]
            py = poly[1::2]
            annos[i]['bbox'] = [np.min(px), np.min(py), np.max(px), np.max(py)]
            annos[i]['bbox_mode'] = BoxMode.XYXY_ABS
            annos[i]['category_id'] -= 1

        record['annotations'] = annos
        dataset_dicts.append(record)
    return dataset_dicts

for d in ['train', 'val']:
    DatasetCatalog.register('publaynet_'+d, lambda d=d: get_publaynet_dicts('data/publaynet/', split=d))
    MetadataCatalog.get('publaynet_'+d).set(thing_classes=['text', 'title', 'list', 'table', 'figure'])
    MetadataCatalog.get('publaynet_'+d).set(evaluator_type='coco')
publaynet_metadata = MetadataCatalog.get('publaynet_train')

if __name__ == '__main__':
    dataset_dicts = get_publaynet_dicts('data/publaynet', 'train')
    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:,:,::-1], metadata=publaynet_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        temp = d
        cv2.imwrite(f'{i}.png', out.get_image()[:,:,::-1])

