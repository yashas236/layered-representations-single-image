"""End-to-end: image -> semantic layers (instance or semantic segmentation) with
depth ordering -> RGBA layer stack + (optional) per-layer albedo/shading split.
"""
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from depth import DepthEstimator
from intrinsic import split_albedo_shading
from layers import build_layers, composite
from segmentation import InstanceSegmenter, SemanticSegmenter


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_on_image(image_rgb_uint8, segmenter, depth_estimator, out_dir, name, do_intrinsic=False):
    os.makedirs(out_dir, exist_ok=True)

    instances = segmenter(image_rgb_uint8)
    disparity = depth_estimator(image_rgb_uint8)
    layer_list = build_layers(image_rgb_uint8, instances, disparity)
    recon = composite(layer_list, image_rgb_uint8.shape)

    psnr = peak_signal_noise_ratio(image_rgb_uint8, recon)
    ssim = structural_similarity(image_rgb_uint8, recon, channel_axis=2)

    for i, layer in enumerate(layer_list):
        Image.fromarray(layer["rgba"], mode="RGBA").save(
            os.path.join(out_dir, f"{name}_layer{i:02d}_{layer['group']}_{layer['class_name']}.png")
        )
        if do_intrinsic:
            albedo, shading = split_albedo_shading(layer["rgba"])
            Image.fromarray(albedo, mode="RGBA").save(os.path.join(out_dir, f"{name}_layer{i:02d}_albedo.png"))
            Image.fromarray(shading, mode="RGBA").save(os.path.join(out_dir, f"{name}_layer{i:02d}_shading.png"))

    Image.fromarray(image_rgb_uint8).save(os.path.join(out_dir, f"{name}_input.png"))
    Image.fromarray(recon).save(os.path.join(out_dir, f"{name}_recomposited.png"))

    return {
        "name": name,
        "n_layers": len(layer_list),
        "groups": [l["group"] for l in layer_list],
        "classes": [l["class_name"] for l in layer_list],
        "psnr": float(psnr),
        "ssim": float(ssim),
        "disparity": disparity,
        "layers": layer_list,
        "recon": recon,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--backend", choices=["instance", "semantic"], default="instance")
    parser.add_argument("--out_dir", type=str, default="../outputs")
    parser.add_argument("--intrinsic", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    segmenter = InstanceSegmenter(device) if args.backend == "instance" else SemanticSegmenter(device)
    depth_estimator = DepthEstimator(device)

    image = np.array(Image.open(args.image_path).convert("RGB"))
    name = os.path.splitext(os.path.basename(args.image_path))[0]
    result = run_on_image(image, segmenter, depth_estimator, args.out_dir, name, do_intrinsic=args.intrinsic)
    print(f"{name}: {result['n_layers']} layers {result['classes']} PSNR={result['psnr']:.2f} SSIM={result['ssim']:.4f}")
