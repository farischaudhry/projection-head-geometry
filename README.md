# The Geometry of Projection Heads: Conditioning, Invariance, and Collapse

This repository contains experimental validation for geometric analysis of projection heads in self-supervised learning frameworks (SimCLR, SimSiam, VICReg, Barlow Twins, DINO).

## Installation

Requires Python 3.10 or higher with CUDA support recommended.

```bash
pip install uv
uv sync
```

This will install all required packages specified in `pyproject.toml`.

## Project Structure

```plaintext
projection-head-geometry/
├── experiments.py              # Main experiment runner
├── plot_results.py             # Visualization script
├── head_depth_ablations.py     # Geometry ablations for head depth
├── hessian_tracker.py          # Tracker for residuals, min eigenvalue, and cond. number
├── results/
│   └── {dataset}/{architecture}/  # Results organized by dataset and architecture
│       ├── collapse_results.npy
│       ├── guillotine_results.npy
│       ├── curvature_results.npy
│       ├── orbit_visualization.npy
│       ├── fig1_collapse_instability.png
│       ├── fig1_swish_robustness.png
│       ├── fig1_relu_gap.png
│       ├── fig1_gradient_norms.png
│       ├── fig1_condition_numbers.png
│       ├── fig1_timescale_separation.png
│       ├── fig2_geometric_mechanisms.png
│       └── fig3_orbit_visualization.png
├── pretrained.ipynb            # Script for geometric analysis of pretrained models
├── pretrained/                 # Folder for pretrained model checkpoints
└── logs/                       # Timestamped experiment logs
```

## Quick Start

### Running Experiments

Run all experiments on CIFAR-10 with ResNet-18 (default):
```bash
uv run experiments.py
```

For CIFAR-100, ResNet-18:
```bash
uv run experiments.py --dataset cifar100
```

Custom training configuration (CIFAR-10, ViT-T):
```bash
uv run experiments.py \
  --dataset cifar10 \
  --architecture vit_tiny \
  --num_epochs_collapse 20 \
  --num_epochs_guillotine 50 \
  --batch_size 512
```

### Generating Figures

After running experiments, generate final figures e.g.,
```bash
uv run plot_results.py --dataset cifar10 --architecture resnet18
```

## Experiments

### Experiment 1: Collapse Instability

Curvature destabilizes collapse dynamics. Tests projection head activation functions (Linear, ReLU, GELU, Swish), plus batch-norm and learning-rate ablations, initialized near a collapsed state (i.e., low representation variance). Tracks representation standard deviation, gradient norms, and covariance conditioning over training epochs to measure escape dynamics and timescale separation.

Linear heads remain collapsed while nonlinear activations (GELU, Swish) escape due to induced manifold curvature.

**Output**: 
- `collapse_results.npy`: Variance, gradient, and conditioning trajectories per run
- `fig1_collapse_instability.png`: Main multi-seed comparison plot
- `fig1_swish_robustness.png`: Swish BN/LR ablations
- `fig1_relu_gap.png`: ReLU BN/LR ablations
- `fig1_gradient_norms.png`: Projection head residual gradient norms
- `fig1_condition_numbers.png`: Covariance condition numbers
- `fig1_timescale_separation.png`: Head vs backbone relative updates

### Experiment 2a: Guillotine Effect

Metric singularity via geometric transformation. Pretrains a SimCLR model, then probe rotation prediction accuracy using both:
1. **Linear probes** - measure linearly accessible information
2. **MLP probes** - measure total recoverable information

Tests both backbone representations $z$ and projection head outputs $h(z)$. Generally, a large performance gap is noted between head and backbone (in both linear and MLP probes).

**Output**:
- `guillotine_results.npy`: Probe accuracies and gaps
- Left panel of `fig2_geometric_mechanisms.png`

### Experiment 2b: Manifold Curvature

Direct measurement of geometric warping. Sweeps rotation angles 0° → 45° and computes local curvature of the resulting trajectory in representation space using finite difference approximation:

$$\kappa \approx \left\| \frac{d^2r}{d\theta^2} \right\|$$

Compares curvature between backbone and projection head over multiple random seeds. Projection head exhibits significantly higher curvature than backbone, a result hypothesised to be due to manifold warping as a way of annihilating augmentation orbits.

**Output**:
- `curvature_results.npy`: Curvature statistics with confidence intervals
- Right panel of `fig2_geometric_mechanisms.png`

### Experiment 2c: Orbit Visualization

Visualization of augmentation orbit collapse via metric singularity. Samples images from multiple classes and applies continuous rotation transformations, collecting representations from both backbone and projection head. Computes high-dimensional geometric metrics:

- **Mean orbit spread**: Variance of augmentation-induced representations
- **Intra-orbit distance (D_intra)**: Average pairwise distance within augmentation orbits
- **Inter-class distance (D_inter)**: Average distance between class centroids
- **Class/orbit ratio**: Signal-to-noise ratio measuring semantic separation relative to augmentation variance

Provides direct empirical validation of metric singularity theory by quantifying how the projection head collapses augmentation manifolds while preserving semantic structure.

**Output**:
- `orbit_visualization.npy`: High-dimensional orbit data and geometric metrics
- `fig3_orbit_visualization.png`: PCA projection showing orbit collapse with master metrics table printed to console

## Configuration

Default experiment settings (see `ExperimentConfig` in `experiments.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_seeds` | 3 | Random seeds for confidence intervals |
| `batch_size` | 512 | Training batch size |
| `num_epochs_collapse` | 50 | Epochs for collapse experiment |
| `num_epochs_guillotine` | 50 | SimCLR pretraining epochs |
| `exp2_probe_epochs` | 5 | Probe training epochs |
| `lr_collapse` | 0.05 | Learning rate for SimSiam |
| `lr_guillotine` | 1e-3 | Learning rate for SimCLR |

## Additional Experiments

### Head Depth Ablation

Runs a SimCLR pretraining stage with a flexible projection head depth, then evaluates rotation probes, curvature, and orbit geometry.

```bash
uv run head_depth_ablations.py --task geometry --num_layers 2
```

To generate orbit visualizations and metric tables for all depths:

```bash
uv run head_depth_ablations.py --task plot
```

**Output**:
- `geometric_ablation_depth_{N}.npz`: Probe accuracy, curvature, and orbit data
- `fig3_orbit_visualization_depth_{N}.png`: Orbit plots per depth

### Hessian Tracker

Tracks the minimum Hessian eigenvalue (via HVPs), representation variance, and covariance condition number during SimSiam-style training.

```bash
uv run hessian_tracker.py --mode track --init collapsed --activation swish --epochs 100 --plot_after_track
```

To generate the composite plot from saved runs:

```bash
uv run hessian_tracker.py --mode plot --plot_type composite --dir <dir_with_raw_data>
```

**Output**:
- `raw_data_{init}_{activation}.npz`: Tracked metrics over epochs
- `hessian_sig_{init}_{activation}.png`: Min-eigenvalue and variance plot
- `cond_num_{init}_{activation}.png`: Condition-number plot
- `hessian_tracker.png`: Composite figure (plot mode)

## Tests on Pretrained Checkpoints

This project uses the pretrained checkpoints (backbone + projection head) for geometric analysis of larger, pretrained checkpoints.
Below is a list of checkpoints that are released with both the backbone and projection head (note that easy-to-load models on torch.hub have the head discarded).

### VICReg

- Download: [VICReg ResNet-50 full checkpoint](https://dl.fbaipublicfiles.com/vicreg/resnet50_fullckpt.pth)  
- Source: [facebookresearch/vicreg](https://github.com/facebookresearch/vicreg)
- Place the downloaded file in folder: `pretrained/`


### Dino

- Download [Dino ResNet-50 full checkpoint](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth)
- Download [Dino ViT-S/16](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth)
- Source: [facebookresearch/dino](https://github.com/facebookresearch/dino)
- Place the downloaded files in folder: `pretrained/`


### Barlow Twins

- Download [Barlow Twins ResNet-50 full checkpoint](https://dl.fbaipublicfiles.com/barlowtwins/ljng/checkpoint.pth)
- Source: [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)
- Place the downloaded files in folder: `pretrained/`


<!-- ### SimCLR

- Download [SimCLR]
- Source: [google-research/simclr](https://github.com/google-research/simclr)
- Place the downloaded files in folder: `pretrained/` -->
