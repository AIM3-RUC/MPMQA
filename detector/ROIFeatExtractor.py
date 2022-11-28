import numpy as np
import cv2
import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import ImageList, Boxes
from detectron2.checkpoint import DetectionCheckpointer
import sys
sys.path.append('detector')
from bua.d2 import add_attribute_config


class ROIFeatExtractor(nn.Module):
    def __init__(self, model_cfg, weights, bua=False):
        super().__init__()
        self.cfg = get_cfg()
        self.bua = bua
        if self.bua:
            add_attribute_config(self.cfg)
        self.cfg.merge_from_file(model_cfg)
        self.cfg.MODEL.WEIGHTS = weights

        self.model = build_model(self.cfg)

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
    
    def preprocess_images(self, images):
        images = [img.to(self.model.device) for img in images]
        # import pdb;pdb.set_trace()
        images = [img.permute(2, 0, 1) for img in images]
        images = [(x - self.model.pixel_mean) / self.model.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)
        return images
    
    def convert_bbox(self, bboxes):
        bboxes = [Boxes(b.to(self.model.device)) for b in bboxes]
        return bboxes

    def roi_align(self, grid_features, bboxes):
        nbbs = [len(b) for b in bboxes]
        if self.bua:
            box_features = self.model.roi_heads._shared_roi_transform(
                [grid_features[f] for f in self.model.roi_heads.in_features], 
                bboxes)
            box_features = box_features.mean(dim=[2, 3])
        else:
            grid_features = [grid_features[f] for f in self.model.roi_heads.box_in_features]
            box_features = self.model.roi_heads.box_pooler(grid_features, bboxes)
            box_features = self.model.roi_heads.box_head(box_features)
        box_features = box_features.split(nbbs)
        return box_features
    


    def forward(self, images, bboxes):
        """
        args:
            image - BGR image list [H x W x C, ]
            bboxes - Boxes of each image, list [N x 4,]
        
        """

        images = self.preprocess_images(images)
        bboxes = self.convert_bbox(bboxes)
        grid_features = self.model.backbone(images.tensor)
        roi_features = self.roi_align(grid_features, bboxes)
        return roi_features
    
    def predict(self, roi_features):
        predictions = []
        logits = []
        for f in roi_features:
            logits.append(
                self.model.roi_heads.box_predictor(f)[0]#.argmax(dim=-1)
            )
        for l in logits:
            predictions.append(
                l[:, :-1].argmax(dim=-1)
            )
        return predictions, logits


if __name__ == '__main__':
    # extractor = ROIFeatExtrator('expr/vg-rcnn-3x/config.yaml', 'expr/vg-rcnn-3x/output/ckpts/model_final.pth')
    extractor = ROIFeatExtrator('expr/vg-bua/config.yaml', 'pretrained/bua-d2-frcn-r101.pth', bua=True)
    extractor.eval()
    images = [torch.from_numpy(cv2.imread('2344092.jpg'))]
    bboxes = [torch.from_numpy(np.load('expr/vg-rcnn-3x/output/result_samples/2344092/boxes.npy'))]
    classes = [torch.from_numpy(np.load('expr/vg-rcnn-3x/output/result_samples/2344092/classes.npy'))]

    images.append(torch.from_numpy(cv2.imread('data/VisualGenome/VG_100K/2368275.jpg')))
    bboxes.append(torch.from_numpy(np.load('expr/vg-rcnn-3x/output/result_samples/2368275/boxes.npy')))
    classes.append(torch.from_numpy(np.load('expr/vg-rcnn-3x/output/result_samples/2368275/classes.npy')))
    
    with torch.no_grad():
        roi_features = extractor(images, bboxes)
        predictions, logits = extractor.predict(roi_features)
        # predictions = predictions.detach().cpu()
        import pdb;pdb.set_trace()
        for c1, p1 in zip(classes, predictions):
            if all(c1 == p1.detach().cpu()) != True:
                import pdb;pdb.set_trace()
