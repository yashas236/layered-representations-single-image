"""Grid figure: input, depth map, segmentation overlay, individual RGBA layers, recomposite."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depth import normalize_depth


def make_figure(image, result, out_path):
    layers = result["layers"]
    n_layers = len(layers)
    n_cols = max(4, n_layers)
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 6.5))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(normalize_depth(result["disparity"]), cmap="inferno")
    axes[0, 1].set_title("MiDaS disparity\n(brighter = nearer)")
    axes[0, 2].imshow(result["recon"])
    axes[0, 2].set_title(f"Recomposited\nPSNR={result['psnr']:.1f} SSIM={result['ssim']:.3f}")
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0, 0.5,
        "Layers (near→far):\n" + "\n".join(f"{i}: {g}/{c}" for i, (g, c) in enumerate(zip(result["groups"], result["classes"]))),
        fontsize=9, va="center",
    )
    for c in range(4, n_cols):
        axes[0, c].axis("off")

    checker = np.indices((image.shape[0] // 8 + 1, image.shape[1] // 8 + 1)).sum(axis=0) % 2
    checker = np.kron(checker, np.ones((8, 8)))[:image.shape[0], :image.shape[1]]
    checker_rgb = np.stack([checker] * 3, axis=-1) * 60 + 195

    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_layers:
            layer = layers[i]
            rgba = layer["rgba"].astype(np.float32)
            alpha = rgba[..., 3:4] / 255.0
            disp = checker_rgb * (1 - alpha) + rgba[..., :3] * alpha
            ax.imshow(disp.astype(np.uint8))
            ax.set_title(f"L{i}: {layer['group']}\n{layer['class_name']}", fontsize=9)
        ax.axis("off")

    for ax in axes[0, :4]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
