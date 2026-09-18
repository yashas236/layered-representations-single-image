# Layered Representations from a Single Image

Optional bonus project for **BSDA5006 – Deep Learning for Computer Vision** (IIT Hyderabad),
per the released project statement: given a single RGB image, produce a stack of RGBA
layers with (a) semantic grouping, (b) depth order (near→far), and (c) an optional
per-layer intrinsic albedo/shading split.

See [`report/report.md`](report/report.md) for the literature review, method, benchmarking, and results.

## Approach

Rather than training a new model from scratch under a tight time budget, this project
composes three well-established, pretrained building blocks into a working layering
pipeline, and benchmarks a genuine design choice within it:

1. **Semantic grouping** — two interchangeable, pretrained segmentation backbones are
   implemented and benchmarked against each other for this task:
   - **Mask R-CNN** (instance segmentation, COCO-pretrained): one layer per *object instance*.
   - **DeepLabV3** (semantic segmentation, COCO-pretrained on VOC classes): one layer per *class*, merging same-class instances.
   Detected classes are mapped into the project's requested broad groups (people, animals,
   vehicles, furniture; everything else falls into "other", and unclaimed pixels form "background").
2. **Depth ordering** — **MiDaS** (small variant) monocular depth estimation gives a
   per-pixel disparity map; each layer's mean disparity within its mask determines its
   near→far position in the stack.
3. **Intrinsic albedo/shading split (stretch goal)** — a classical Retinex-style heuristic
   (low-pass luminance as shading, residual as albedo), applied per layer within its alpha
   mask. This is explicitly a non-learned approximation, not a trained intrinsic-decomposition
   network — see the report's limitations section for why, and what a learned version would need.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pretrained weights (Mask R-CNN, DeepLabV3, MiDaS) download automatically on first run via
`torch.hub`/`torchvision`. Sample images are the ones bundled with `scikit-image`
(astronaut, cat, coffee cup, motorcycle) — no external dataset download is required to
reproduce the benchmark.

## Project structure

```
src/
  segmentation.py   Mask R-CNN (instance) + DeepLabV3 (semantic) wrappers, COCO/VOC -> group mapping
  depth.py          MiDaS monocular depth estimator
  layers.py         Builds depth-ordered RGBA layers from masks + a depth map, and re-composites them
  intrinsic.py       Classical albedo/shading split (stretch goal)
  pipeline.py        End-to-end single-image pipeline + CLI
  benchmark.py       Runs the pipeline over sample images x both segmentation backbones
  visualize.py       Per-image grid figure (input, depth, layers, recomposite)
outputs/             Per-image RGBA layer stacks + recomposited images (generated)
figures/             Grid visualizations per image/backend (generated)
logs/                benchmark_table.md / benchmark_results.json (generated)
report/report.md     Literature review, method, benchmarking, results, limitations
```

## Running it

On your own image:

```bash
cd src
python pipeline.py /path/to/image.jpg --backend instance --intrinsic
```

Outputs land in `outputs/`: one RGBA PNG per layer (named by group/class), the input, and
the recomposited image.

Full benchmark (both backbones x all sample images, regenerates everything in `outputs/`,
`figures/`, `logs/`):

```bash
cd src
python benchmark.py
```
