# 360-SL-Metasurface

A research codebase for 360-degree structured light with learned metasurfaces.

## What This Project Does

1. Learns a metasurface phase map for structured-light projection.
2. Simulates differentiable fisheye image formation.
3. Reconstructs depth with stereo matching from synthetic active illumination.
4. Evaluates depth metrics and visualizes phase/metric results.

## Repository Layout

- `model/`: metasurface, stereo matching, and end-to-end model code
- `Image_formation/`: differentiable rendering pipeline
- `dataset/`: dataset loader and synthetic data generation scripts
- `utils/`: argument parsing, camera models, network utilities, rendering utilities
- `scripts/`: training, evaluation, comparison, and plotting entrypoints

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate metaPolka
```

## Data Setup

Default dataset root is `./11518075/`.

By default:
- train split: `./11518075/train`
- validation/test split: `./11518075/test`

These defaults are defined in `utils/ArgParser.py`.

## Main Executable Scripts

### Training

```bash
python -m scripts.train_baseline
python -m scripts.train_enhanced
```

### Depth Evaluation

```bash
python -m scripts.eval_depth
```

### Metric Evaluation

```bash
python -m scripts.eval_metrics_cam1
python -m scripts.eval_metrics_cam1_legacy
python -m scripts.eval_metrics_cam1_plus
python -m scripts.eval_metrics_cam_both
```

### Comparison and Visualization

```bash
python -m scripts.compare_loss
python -m scripts.compare_phase_maps
python -m scripts.select_best_checkpoint
python -m scripts.plot_radar_metrics
python -m scripts.plot_phase_map
python -m scripts.plot_phase_map_with_circle
python -m scripts.plot_phase_map_with_circle_v2
```

### Dataset Generation

```bash
python dataset/dataset_generator.py
```

## References

- Project Page: https://eschoi.com/360-SL-Metasurface/
- Paper: https://www.nature.com/articles/s41566-024-01450-x
- ArXiv: https://arxiv.org/abs/2306.13361
