"""Classical (non-learned) intrinsic albedo/shading split — the stretch goal.

This is a fast Retinex-style heuristic, not a trained intrinsic-decomposition
network: shading is approximated as a heavily low-pass-filtered luminance
channel (the classical assumption that illumination varies smoothly while
albedo/texture carries the high-frequency detail), and albedo is the residual.
It is included per-layer, applied only within each layer's alpha mask, and is
explicitly flagged in the report as an approximation rather than a learned
result, given the project's time budget.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def split_albedo_shading(rgba, sigma=15):
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3] > 0

    luminance = rgb.mean(axis=2)
    shading = gaussian_filter(luminance, sigma=sigma)
    shading = np.clip(shading, 0.05, 1.0)  # avoid divide-by-zero in flat/transparent regions

    albedo = rgb / shading[..., None]
    albedo = np.clip(albedo, 0.0, 1.0)

    shading_rgba = np.zeros_like(rgba)
    shading_rgba[..., :3] = (np.repeat(shading[..., None], 3, axis=2) * 255).astype(np.uint8)
    shading_rgba[..., 3] = rgba[..., 3]

    albedo_rgba = np.zeros_like(rgba)
    albedo_rgba[..., :3] = (albedo * 255).astype(np.uint8)
    albedo_rgba[..., 3] = rgba[..., 3]

    return albedo_rgba, shading_rgba
