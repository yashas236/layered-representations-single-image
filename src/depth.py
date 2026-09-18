"""Monocular depth estimation (MiDaS small) used to order layers near -> far."""
import os

import numpy as np
import torch


def _pre_trust_midas_repos():
    """MiDaS's own hubconf.py internally calls torch.hub.load('rwightman/gen-efficientnet-
    pytorch', ...) without forwarding our trust_repo=True, so torch.hub still prompts
    interactively for that nested dependency on a first run. Pre-populating torch.hub's
    trusted_list (the same list `trust_repo=True` itself writes to) makes first-run setup
    non-interactive, since these are exactly the repos this project intentionally loads."""
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)
    trusted_path = os.path.join(hub_dir, "trusted_list")
    needed = {"intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch"}
    existing = set()
    if os.path.exists(trusted_path):
        existing = {line.strip() for line in open(trusted_path)}
    with open(trusted_path, "a") as f:
        for repo in needed - existing:
            f.write(repo + "\n")


class DepthEstimator:
    def __init__(self, device):
        _pre_trust_midas_repos()
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
