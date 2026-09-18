"""Demonstrates the project's stated motivation (parallax animation) directly: shift each
RGBA layer horizontally by an amount proportional to its depth (nearer layers move more,
as in real parallax from a small camera translation), then recomposite. Because layers
have no occlusion inpainting (Section 6 of the report), shifting reveals disocclusion holes
where a foreground layer moved away from background that was never captured -- this
function makes that limitation directly visible rather than just asserted.
"""
import numpy as np


def shift_layer(rgba, dx):
    h, w = rgba.shape[:2]
    shifted = np.zeros_like(rgba)
    dx = int(np.clip(dx, -w, w))
    if dx == 0:
        return rgba.copy()
    if dx > 0:
        shifted[:, dx:] = rgba[:, :w - dx]
    else:
        shifted[:, :w + dx] = rgba[:, -dx:]
    return shifted


def render_parallax(layers, canvas_shape, max_shift_px=40):
    """Returns (shifted_composite_rgb, hole_mask) where hole_mask marks pixels no
    layer covers after shifting -- i.e. disocclusion holes from missing inpainting."""
    disparities = np.array([l["mean_disparity"] for l in layers], dtype=np.float32)
    span = disparities.max() - disparities.min()
    d_norm = (disparities - disparities.min()) / (span + 1e-8)

    h, w = canvas_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.float32)

    order = np.argsort(disparities)  # far (low disparity) -> near (high disparity)
    for idx in order:
        layer = layers[idx]
        dx = d_norm[idx] * max_shift_px  # nearer (higher disparity) shifts more
        shifted = shift_layer(layer["rgba"], dx)
        alpha = shifted[..., 3:4].astype(np.float32) / 255.0
        canvas = canvas * (1 - alpha) + shifted[..., :3].astype(np.float32) * alpha
        coverage = np.maximum(coverage, alpha[..., 0])

    hole_mask = coverage < 0.5
    return canvas.astype(np.uint8), hole_mask
