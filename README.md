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
- top-level scripts: training, evaluation, comparison, and plotting entrypoints

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
python train_baseline.py
python train_enhanced.py
```

### Depth Evaluation

```bash
python eval_depth.py
```

### Metric Evaluation

```bash
python eval_metrics_cam1.py
python eval_metrics_cam1_legacy.py
python eval_metrics_cam1_plus.py
python eval_metrics_cam_both.py
```

### Comparison and Visualization

```bash
python compare_loss.py
python compare_phase_maps.py
python select_best_checkpoint.py
python plot_radar_metrics.py
python plot_phase_map.py
python plot_phase_map_with_circle.py
python plot_phase_map_with_circle_v2.py
```

### Dataset Generation

```bash
python dataset/dataset_generator.py
```

## References

- Project Page: https://eschoi.com/360-SL-Metasurface/
- Paper: https://www.nature.com/articles/s41566-024-01450-x
- ArXiv: https://arxiv.org/abs/2306.13361
