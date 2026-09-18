"""Benchmark: run the layering pipeline on a handful of sample images (bundled with
scikit-image, so no dataset download is needed) using both segmentation backbones
(Mask R-CNN instance segmentation vs. DeepLabV3 semantic segmentation), and compare
layer counts and reconstruction quality (PSNR/SSIM of the recomposited image).
"""
import json
import os

import numpy as np
from PIL import Image
from skimage import data as skdata

from depth import DepthEstimator
from pipeline import get_device, run_on_image
from segmentation import InstanceSegmenter, SemanticSegmenter
from visualize import make_figure


def load_sample_images():
    samples = {
        "astronaut": skdata.astronaut(),   # person
        "chelsea": skdata.chelsea(),       # cat (animal)
        "coffee": skdata.coffee(),         # mug / table (furniture-ish still life)
    }
    try:
        left, _ = skdata.stereo_motorcycle()
        samples["motorcycle"] = left[..., :3]
    except Exception:
        pass
    return samples


def main():
    device = get_device()
    print(f"Using device: {device}")

    depth_estimator = DepthEstimator(device)
    instance_seg = InstanceSegmenter(device)
    semantic_seg = SemanticSegmenter(device)

    samples = load_sample_images()
    results = []

    for name, image in samples.items():
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)

        for backend_name, segmenter in [("instance_maskrcnn", instance_seg), ("semantic_deeplabv3", semantic_seg)]:
            out_dir = f"../outputs/{backend_name}"
            res = run_on_image(image, segmenter, depth_estimator, out_dir, name, do_intrinsic=(backend_name == "instance_maskrcnn"))
            make_figure(image, res, f"../figures/{name}_{backend_name}.png")
            print(f"[{backend_name}] {name}: {res['n_layers']} layers {res['classes']} PSNR={res['psnr']:.2f} SSIM={res['ssim']:.4f}")
            results.append({
                "image": name, "backend": backend_name, "n_layers": res["n_layers"],
                "groups": res["groups"], "classes": res["classes"],
                "psnr": res["psnr"], "ssim": res["ssim"],
            })

    os.makedirs("../logs", exist_ok=True)
    with open("../logs/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # markdown summary table
    lines = ["| Image | Backend | # Layers | Groups detected | PSNR (dB) | SSIM |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['image']} | {r['backend']} | {r['n_layers']} | {', '.join(sorted(set(r['groups'])))} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open("../logs/benchmark_table.md", "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
