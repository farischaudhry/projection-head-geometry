# The Geometry of Projection Heads: Conditioning, Invariance, and Collapse

This repository contains experimental validation for geometric analysis of projection heads in self-supervised learning frameworks (SimCLR, SimSiam).

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
├── results/
│   └── {dataset}/             # Results organized by dataset
│       ├── collapse_results.npy
│       ├── guillotine_results.npy
│       ├── curvature_results.npy
│       ├── fig1_collapse_instability.png
│       └── fig2_geometric_mechanisms.png
└── logs/                       # Timestamped experiment logs
```

## Quick Start

### Running Experiments

Run all experiments on CIFAR-10 (default):
```bash
uv run experiments.py
```

For CIFAR-100:
```bash
uv run experiments.py --dataset cifar100
```

Custom training configuration:
```bash
uv run experiments.py \
  --dataset cifar10 \
  --num_epochs_collapse 20 \
  --num_epochs_guillotine 50 \
  --batch_size 512
```

### Generating Figures

After running experiments, generate final figures:
```bash
uv run plot_results.py --dataset cifar10
```

## Experiments

### Experiment 1: Collapse Instability

Curvature destabilizes collapse dynamics. Tests four projection head activation functions (Linear, ReLU, GELU, Swish) initialized near a collapsed state (i.e., low representation variance). Tracks representation standard deviation over training epochs to measure escape dynamics.

Linear heads remain collapsed while nonlinear activations (GELU, Swish) escape due to induced manifold curvature.

**Output**: 
- `collapse_results.npy`: Variance trajectories per activation type
- `fig1_collapse_instability.png`: Multi-seed comparison plot

### Experiment 2a: Guillotine Effect

Metric singularity via geometric transformation. Pretrains a SimCLR model, then probe rotation prediction accuracy using both:
1. **Linear probes** - measure linearly accessible information
2. **MLP probes** - measure total recoverable information

Tests both backbone representations $z$ and projection head outputs $h(z)$. Genearlly, a large performance gap is noted between head and backbone (in both linear and MLP probes).

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

## Configuration

Default experiment settings (see `ExperimentConfig` in `experiments.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_seeds` | 3 | Random seeds for confidence intervals |
| `batch_size` | 512 | Training batch size |
| `num_epochs_collapse` | 10 | Epochs for collapse experiment |
| `num_epochs_guillotine` | 20 | SimCLR pretraining epochs |
| `exp2_probe_epochs` | 5 | Probe training epochs |
| `lr_collapse` | 0.05 | Learning rate for SimSiam |
| `lr_guillotine` | 1e-3 | Learning rate for SimCLR |
