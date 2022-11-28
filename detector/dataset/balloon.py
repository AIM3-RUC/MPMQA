import os
import json
import cv2
import numpy as np

from detectron2.structures import BoxMode

# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        filename = os.path.join(img_dir, v['filename'])
        height, width = cv2.imread(filename).shape[:2]
        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width

        annos = v['regions']
        objs = []

        for _, anno in annos.items():
            assert not anno['region_attributes']
            anno = anno['shape_attributes']
            px = anno["all_points_x"]
            py = anno['all_points_y']
            poly = [(x+0.5, y+0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                'bbox': [np.min(px), np.min(py), np.max(px), np.max(py)],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': [poly],
                'category_id': 0
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ['train', 'val']:
    DatasetCatalog.register('balloon_'+d, lambda d=d: get_balloon_dicts('data/balloon/'+d))
    MetadataCatalog.get('balloon_'+d).set(thing_classes=['balloon'])
balloon_metadata = MetadataCatalog.get('balloon_train')

if __name__ == '__main__':
    dataset_dicts = get_balloon_dicts('data/balloon/train')
    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:,:,::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        temp = d
        cv2.imwrite(f'{i}.png', out.get_image()[:,:,::-1])

