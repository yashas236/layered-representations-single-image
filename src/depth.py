"""Monocular depth estimation (MiDaS small) used to order layers near -> far."""
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform

    @torch.no_grad()
    def __call__(self, image_rgb_uint8):
        """Returns a per-pixel relative depth map, larger value = farther away
        is NOT guaranteed by MiDaS (it predicts inverse depth / disparity, larger
        = closer). We keep MiDaS's raw convention (higher = closer) and sort
        accordingly wherever this is consumed."""
        inp = self.transform(image_rgb_uint8).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=image_rgb_uint8.shape[:2], mode="bicubic", align_corners=False,
        ).squeeze()
        return pred.cpu().numpy()


def normalize_depth(depth):
    d = depth - depth.min()
    d = d / (d.max() + 1e-8)
    return d
