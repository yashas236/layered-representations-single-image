"""Semantic grouping via pretrained instance/semantic segmentation.

Two interchangeable backbones are provided so the project can benchmark them
against each other for the layering task:
  - Mask R-CNN (instance segmentation): one mask per object instance.
  - DeepLabV3 (semantic segmentation): one mask per class, instances of the
    same class are merged into a single layer.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# Broad semantic groups the project statement asks for, built from COCO category names.
GROUP_KEYWORDS = {
    "people": {"person"},
    "animals": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
    "vehicles": {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"},
    "furniture": {"chair", "couch", "bed", "dining table", "toilet", "bench"},
}


def group_for_class(class_name):
    for group, names in GROUP_KEYWORDS.items():
        if class_name in names:
            return group
    return "other"


class InstanceSegmenter:
    """Mask R-CNN: returns one (mask, class_name, group, score) per detected instance."""

    def __init__(self, device, score_thresh=0.6):
        self.device = device
        self.score_thresh = score_thresh
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        self.model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()

    @torch.no_grad()
    def __call__(self, image_rgb_uint8):
        x = TF.to_tensor(image_rgb_uint8).to(self.device)
        out = self.model([x])[0]
        instances = []
        for mask, label, score in zip(out["masks"], out["labels"], out["scores"]):
            if score.item() < self.score_thresh:
                continue
            class_name = self.categories[label.item()]
            binary_mask = (mask[0].cpu().numpy() > 0.5)
            if binary_mask.sum() < 50:  # drop specks
                continue
            instances.append({
                "mask": binary_mask,
                "class_name": class_name,
                "group": group_for_class(class_name),
                "score": score.item(),
            })
        return instances


class SemanticSegmenter:
    """DeepLabV3: returns one (mask, class_name, group) per class present in the image."""

    def __init__(self, device, min_area=200):
        self.device = device
        self.min_area = min_area
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.categories = weights.meta["categories"]  # Pascal VOC 21 classes
        self.model = deeplabv3_resnet50(weights=weights).to(device).eval()

    @torch.no_grad()
    def __call__(self, image_rgb_uint8):
        x = TF.to_tensor(image_rgb_uint8).unsqueeze(0).to(self.device)
        out = self.model(x)["out"][0]
        pred = out.argmax(0).cpu().numpy()
        instances = []
        for class_idx in np.unique(pred):
            if class_idx == 0:  # background class in VOC
                continue
            binary_mask = pred == class_idx
            if binary_mask.sum() < self.min_area:
                continue
            class_name = self.categories[class_idx]
            instances.append({
                "mask": binary_mask,
                "class_name": class_name,
                "group": group_for_class(class_name),
                "score": 1.0,
            })
        return instances
