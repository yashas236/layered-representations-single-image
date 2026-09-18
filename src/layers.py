"""Build depth-ordered RGBA layers from an image, its instance masks, and a depth map."""
import numpy as np


def build_layers(image_rgb_uint8, instances, disparity_map):
    """Returns a list of layer dicts, ordered near -> far:
        {"rgba": HxWx4 uint8, "group": str, "class_name": str, "mean_disparity": float}
    MiDaS disparity: higher value = closer to the camera, so we sort descending.
    Any pixel not covered by an instance mask becomes the single "background" layer.
    """
    h, w = disparity_map.shape
    covered = np.zeros((h, w), dtype=bool)
    layers = []

    for inst in instances:
        mask = inst["mask"]
        covered |= mask
        mean_disp = float(disparity_map[mask].mean()) if mask.any() else 0.0
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = image_rgb_uint8
        rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        layers.append({
            "rgba": rgba,
            "group": inst["group"],
            "class_name": inst["class_name"],
            "mean_disparity": mean_disp,
        })

    # background = everything no instance claimed
    bg_mask = ~covered
    bg_mean_disp = float(disparity_map[bg_mask].mean()) if bg_mask.any() else disparity_map.min()
    bg_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    bg_rgba[..., :3] = image_rgb_uint8
    bg_rgba[..., 3] = np.where(bg_mask, 255, 0).astype(np.uint8)
    layers.append({
        "rgba": bg_rgba,
        "group": "background",
        "class_name": "background",
        "mean_disparity": bg_mean_disp,
    })

    # near -> far: higher disparity (MiDaS convention) drawn/listed first
    layers.sort(key=lambda l: l["mean_disparity"], reverse=True)
    return layers


def composite(layers, canvas_shape):
    """Alpha-composite layers in far->near order (reverse of near->far) to reconstruct the image."""
    h, w = canvas_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for layer in reversed(layers):  # far to near
        rgba = layer["rgba"].astype(np.float32)
        alpha = (rgba[..., 3:4] / 255.0)
        canvas = canvas * (1 - alpha) + rgba[..., :3] * alpha
    return canvas.astype(np.uint8)
