import os
import numpy as np
from xml.etree.ElementInclude import default_loader
import cv2
import random
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer
from pkg_resources import DefaultProvider
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from bua.d2 import (
    build_detection_test_loader_with_attributes
)
from bua import add_config
import bua.d2.modeling.roi_heads
from pdfparse.utils import doc2imgs
from evaluation import VGEvaluator

# import dataset.balloon
# from dataset.balloon import balloon_metadata
# from dataset.balloon import get_balloon_dicts
import dataset.publaynet
from dataset.publaynet import publaynet_metadata, get_publaynet_dicts
from dataset.vg import get_vg_dicts, vg_metadata

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

import argparse


def inference_doc(cfg, doc_fp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    imgs = doc2imgs(doc_fp)
    predictor = DefaultPredictor(cfg)
    for i, im in tqdm(enumerate(imgs), total=len(imgs), desc='Detecting'):
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=publaynet_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(out_dir, str(i+1)+'.jpg'), out.get_image()[:,:,::-1])

def inference_dir(cfg, root, out_dir):
    predictor = DefaultPredictor(cfg)
    for img_name in os.listdir(root):
        fp = os.path.join(root, img_name)
        im = cv2.imread(fp)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=publaynet_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(out_dir, img_name), out.get_image()[:, :, ::-1])

def save_instances(instance, save_dir):
    boxes = instance.pred_boxes
    classes = instance.pred_classes
    np.save(save_dir+'/boxes.npy', boxes.tensor.cpu().detach().numpy())
    np.save(save_dir+'/classes.npy', classes.cpu().detach().numpy())

def inference(cfg):
    sample_dir = os.path.join(cfg.OUTPUT_DIR, 'result_samples')
    os.makedirs(sample_dir, exist_ok=True)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_vg_dicts("data/VisualGenome", 'test')
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=vg_metadata, 
                    scale=1, 
                    # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        instance_dir = os.path.join(sample_dir, d['file_name'].split('/')[-1].split('.')[0])
        os.makedirs(instance_dir, exist_ok=True)
        save_instances(outputs['instances'], instance_dir)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(sample_dir, d['file_name'].split('/')[-1]), out.get_image()[:, :, ::-1])

def evaluate(cfg, args):
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    evaluator = VGEvaluator(args.dataset, output_dir='./output', cfg=cfg, distributed=False)
    if cfg.MODEL.ATTRIBUTE_ON:
        val_loader = build_detection_test_loader_with_attributes(cfg, args.dataset)
    else:
        val_loader = build_detection_test_loader(cfg, args.dataset)
    predictor = DefaultPredictor(cfg)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./test//manual_s10.pdf')
    parser.add_argument('--output', default='./test//manual_s10_detect/')
    parser.add_argument('--mode', default='d2')
    parser.add_argument('--thresh', default=0.7, type=float)
    parser.add_argument('--dataset', default='visual_genome_test')

    parser.add_argument('--config', default='./expr/publaynet-rcnn-3x/faster_rcnn_R_50_FPN_3x.yaml')
    parser.add_argument('--model', default='./expr/publaynet-rcnn-3x/output/ckpts/model_final.pth')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    
    cfg = get_cfg()
    args = get_args()
    add_config(args, cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh

    # inference_doc(cfg, args.input, args.output)
    inference(cfg)
    # evaluate(cfg, args)