import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import os
from typing import Literal
from dataclasses import dataclass
import logging
from tqdm import tqdm
from datetime import datetime
import time

# Configure logging
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'logs/{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f'Logging to {log_file}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Plotting style
plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6.18),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
    })


def set_seed(seed: int = 0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    dataset: Literal['cifar10', 'cifar100'] = 'cifar10'
    batch_size: int = 512
    num_epochs_collapse: int = 10
    num_epochs_guillotine: int = 20
    lr_collapse: float = 0.05
    lr_guillotine: float = 1e-3
    momentum: float = 0.9
    collapse_init_scale: float = 0.1
    num_seeds: int = 3
    num_workers: int = 2
    exp2_probe_epochs: int = 5

# =================================
# Shared Components
# =================================

class TwoCropTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class ResNetBackbone(nn.Module):
    """ResNet18 backbone adapted for CIFAR."""
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = output_dim

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.shape[0], -1)


class ProjectionHead(nn.Module):
    """
    Projection head with configurable activation and batch normalization.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        activation: Activation type ('linear', 'relu', 'gelu', 'swish')
        use_bn: Whether to use batch normalization; makes model nonlinear if True
    """
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048, activation='relu', use_bn=True):
        super().__init__()
        layers = []
        
        # Case 1: Strictly Linear (No BN, No Activation)
        if activation == 'linear':
            layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
            self.net = nn.Sequential(*layers)
            return

        # Case 2: Nonlinear
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        if use_bn: 
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())
        else:
            raise ValueError(f'Unknown activation: {activation}')
            
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        if use_bn: 
            layers.append(nn.BatchNorm1d(output_dim))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_dataset(name: Literal['cifar10', 'cifar100'], train=True, transform=None, download=True):
    """Load CIFAR dataset."""
    os.makedirs('./data', exist_ok=True)
    if name.lower() == 'cifar10':
        return torchvision.datasets.CIFAR10(root='./data', train=train, 
                                           transform=transform, download=download)
    elif name.lower() == 'cifar100':
        return torchvision.datasets.CIFAR100(root='./data', train=train, 
                                            transform=transform, download=download)
    else:
        raise ValueError(f'Unknown dataset {name}')


# =================================
# Experiment 1: Collapse Instability
# =================================

def run_collapse_experiment(
    dataset_name: str, 
    activation_type: str, 
    config: ExperimentConfig,
    seeds: list[int] = None
) -> dict:
    """
    Test collapse instability with different activation functions.
    Initialize projector head to pseudo-collapsed state and track variance of representations.
    
    Args:
        dataset_name: Name of dataset ('cifar10' or 'cifar100')
        activation_type: Type of activation ('linear', 'relu', 'gelu', 'swish')
        config: Experiment configuration
        seeds: List of random seeds
        
    Returns:
        Dictionary with variance statistics over training
    """
    if seeds is None:
        seeds = list(range(config.num_seeds))
        
    variances_over_time = []
    
    # SimSiam-style augmentations
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    for seed in seeds:
        set_seed(seed)
        
        train_ds = get_dataset(
            dataset_name, train=True, 
            transform=TwoCropTransform(base_transform)
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers, 
            drop_last=True
        )
        
        # Create models
        backbone = ResNetBackbone().to(device)
        projector = ProjectionHead(
            activation=activation_type, 
            use_bn=False
        ).to(device)
        predictor = ProjectionHead(
            input_dim=2048, 
            hidden_dim=512, 
            output_dim=2048, 
            activation=activation_type, 
            use_bn=False
        ).to(device)
        
        optimizer = optim.SGD(
            list(backbone.parameters()) + 
            list(projector.parameters()) + 
            list(predictor.parameters()), 
            lr=config.lr_collapse, 
            momentum=config.momentum,
            weight_decay=0.0
        )
        
        # Pseudo-Collapsed Initialization
        with torch.no_grad():
            for m in projector.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data *= config.collapse_init_scale
        
        variance_history = []
        logger.info(f"[{dataset_name}] Exp 1: {activation_type} head (Seed {seed})")
        
        for epoch in tqdm(range(config.num_epochs_collapse), desc=f'{activation_type} (seed {seed})', leave=False): 
            backbone.train()
            projector.train()   
            predictor.train()
            epoch_vars = []
            
            for (x1, x2), _ in train_loader:
                x1, x2 = x1.to(device), x2.to(device)
                
                # Forward pass
                z1, z2 = projector(backbone(x1)), projector(backbone(x2))
                p1, p2 = predictor(z1), predictor(z2)
                
                # SimSiam loss
                loss = -(F.cosine_similarity(p1, z2.detach()).mean() + 
                         F.cosine_similarity(p2, z1.detach()).mean()) * 0.5
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track representation variance (representation collapse metric)
                with torch.no_grad():
                    z = torch.cat([z1, z2], dim=0)
                    z = F.normalize(z, dim=1)
                    epoch_vars.append(torch.std(z, dim=0).mean().item())
            
            variance_history.append(np.mean(epoch_vars))
            
        variances_over_time.append(variance_history)
    
    # Return statistics
    variances_np = np.array(variances_over_time)
    return {
        'mean': np.mean(variances_np, axis=0),
        'std': np.std(variances_np, axis=0),
        'raw': variances_np
    }

# =================================
# Experiment 2a: Guillotine Effect (Linear Probing)
# Experiment 2b: Manifold Curvature
# =================================

class RotationDataset(Dataset):
    """Dataset with random rotations for probing."""
    def __init__(self, dataset_name, root='./data', train=True, seed=None):
        self.ds = get_dataset(dataset_name, train=train, transform=None)
        self.to_tensor = transforms.ToTensor()
        if seed is not None:
            rng = random.Random(seed)
            self.rotations = [rng.randint(0, 3) for _ in range(len(self.ds))]
        else:
            self.rotations = [random.randint(0, 3) for _ in range(len(self.ds))]
            
    def __len__(self): 
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        rot_idx = self.rotations[idx]
        img_rotated = transforms.functional.rotate(img, rot_idx * 90)
        return self.to_tensor(img_rotated), rot_idx


def run_guillotine_experiment(
        dataset_name: str, config: ExperimentConfig
    ) -> tuple[float, float, float, float, nn.Module, nn.Module]:
    """
    Test information loss across backbone-head boundary.
    i) Change in probe accuracy from backbone to head
    ii) Nonlinearity gap (MLP vs Linear probe)
    
    Args:
        dataset_name: Name of dataset
        config: Experiment configuration
        
    Returns:
        Tuple of (backbone_accuracy, head_accuracy)
    """
    seed = 0
    set_seed(seed)
    
    # SimCLR-style augmentations
    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ])
    
    train_ds = get_dataset(
        dataset_name, train=True, 
        transform=TwoCropTransform(simclr_transform)
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=256, 
        shuffle=True, 
        drop_last=True, 
        num_workers=config.num_workers
    )
    
    # Create models
    backbone = ResNetBackbone().to(device)
    head = ProjectionHead(activation='relu', use_bn=True).to(device)
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), 
        lr=config.lr_guillotine
    )
    
    logger.info(f'[{dataset_name}] Exp 2a: Pre-training SimCLR ({config.num_epochs_guillotine} Epochs)')
    
    for epoch in tqdm(range(config.num_epochs_guillotine), desc='SimCLR Pre-training'): 
        backbone.train()
        head.train()
        
        for (x1, x2), _ in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            
            # SimCLR loss (NT-Xent)
            z1 = F.normalize(head(backbone(x1)), dim=1)
            z2 = F.normalize(head(backbone(x2)), dim=1)
            
            logits = torch.mm(z1, z2.t()) / 0.1
            labels = torch.arange(x1.size(0)).to(device)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    logger.info(f'[{dataset_name}] Training Rotation Probes')
    backbone.eval()
    head.eval()
    
    # Create rotation datasets
    rot_loader = DataLoader(
        RotationDataset(dataset_name, train=True, seed=seed), 
        batch_size=256, 
        shuffle=True
    )
    test_rot_loader = DataLoader(
        RotationDataset(dataset_name, train=False, seed=seed), 
        batch_size=256, 
        shuffle=False
    )
    
    # Train probes - both linear and nonlinear
    # Linear probes
    probe_z_linear = nn.Linear(512, 4).to(device)
    probe_h_linear = nn.Linear(2048, 4).to(device)
    
    # Nonlinear probes (2-layer MLP)
    probe_z_mlp = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 4)
    ).to(device)
    probe_h_mlp = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 4)
    ).to(device)
    
    opt_z_linear = optim.Adam(probe_z_linear.parameters(), lr=1e-3)
    opt_h_linear = optim.Adam(probe_h_linear.parameters(), lr=1e-3)
    opt_z_mlp = optim.Adam(probe_z_mlp.parameters(), lr=1e-3)
    opt_h_mlp = optim.Adam(probe_h_mlp.parameters(), lr=1e-3)
    
    for epoch in tqdm(range(config.exp2_probe_epochs), desc='Training Probes', leave=False):
        for img, rot_label in rot_loader:
            img, rot_label = img.to(device), rot_label.to(device)
            
            with torch.no_grad():
                feat_z = backbone(img)
                feat_h = head(feat_z)
            
            # Train backbone probes (linear)
            loss_z_linear = F.cross_entropy(probe_z_linear(feat_z), rot_label)
            opt_z_linear.zero_grad()
            loss_z_linear.backward()
            opt_z_linear.step()
            
            # Train head probes (linear)
            loss_h_linear = F.cross_entropy(probe_h_linear(feat_h), rot_label)
            opt_h_linear.zero_grad()
            loss_h_linear.backward()
            opt_h_linear.step()
            
            # Train backbone probes (MLP)
            loss_z_mlp = F.cross_entropy(probe_z_mlp(feat_z), rot_label)
            opt_z_mlp.zero_grad()
            loss_z_mlp.backward()
            opt_z_mlp.step()
            
            # Train head probes (MLP)
            loss_h_mlp = F.cross_entropy(probe_h_mlp(feat_h), rot_label)
            opt_h_mlp.zero_grad()
            loss_h_mlp.backward()
            opt_h_mlp.step()

    def evaluate(probe, use_head):
        """Evaluate probe accuracy."""
        correct = 0
        total = 0
        with torch.no_grad():
            for img, rot_label in test_rot_loader:
                img, rot_label = img.to(device), rot_label.to(device)
                feat = head(backbone(img)) if use_head else backbone(img)
                preds = probe(feat).argmax(dim=1)
                correct += (preds == rot_label).sum().item()
                total += rot_label.size(0)
        return correct / total
    
    # Evaluate all probes
    acc_z_linear = evaluate(probe_z_linear, False)
    acc_h_linear = evaluate(probe_h_linear, True)
    acc_z_mlp = evaluate(probe_z_mlp, False)
    acc_h_mlp = evaluate(probe_h_mlp, True)

    return acc_z_linear, acc_h_linear, acc_z_mlp, acc_h_mlp, backbone, head


def run_curvature_experiment(
    dataset_name: str,
    backbone: nn.Module,
    head: nn.Module,
    config: ExperimentConfig,
    num_samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate local curvature of the rotation manifold.
    
    Measures how warped the manifold is by sweeping through rotations
    and computing the curvature of the resulting trajectory in representation space.
    
    Args:
        dataset_name: Name of dataset
        backbone: Pretrained backbone model
        head: Pretrained projection head
        config: Experiment configuration
        num_samples: Number of images to sample
        
    Returns:
        Tuple of (curvatures_z, curvatures_h) - curvature estimates for backbone and head
    """
    backbone.eval()
    head.eval()
    
    # Load test dataset
    ds = get_dataset(dataset_name, train=False, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    
    curvatures_z = []
    curvatures_h = []
    
    # Rotation angles to sweep (in degrees)
    angles = np.linspace(0, 45, 10)  # 10 points from 0 to 45 degrees
    
    logger.info(f'[{dataset_name}] Exp 2b: Estimating Manifold Curvature')
    logger.info(f'  Sweeping rotations: {angles[0]:.0f}° to {angles[-1]:.0f}° in {len(angles)} steps')
    
    with torch.no_grad():
        for idx, (img, _) in enumerate(tqdm(loader, desc='Computing Curvature', total=num_samples, leave=False)):
            if idx >= num_samples:
                break
                
            img = img.to(device)
            
            # Collect representations at different rotation angles
            reps_z = []
            reps_h = []
            
            for angle in angles:
                img_rot = transforms.functional.rotate(img, float(angle))
                
                # Get representations
                z = F.normalize(backbone(img_rot), dim=1)
                h = F.normalize(head(backbone(img_rot)), dim=1)
                
                reps_z.append(z.cpu().numpy().flatten())
                reps_h.append(h.cpu().numpy().flatten())
            
            # Convert to numpy arrays [num_angles, dim]
            reps_z = np.array(reps_z)
            reps_h = np.array(reps_h)
            
            # Estimate curvature using finite differences
            # Curvature ≈ ||d²r/dt²|| where r(t) is the trajectory
            # Second derivative approximation: (r[i+1] - 2*r[i] + r[i-1]) / dt²
            
            def estimate_curvature(trajectory):
                """Estimate mean curvature of a trajectory."""
                # Use central differences for interior points
                curvs = []
                for i in range(1, len(trajectory) - 1):
                    second_deriv = trajectory[i+1] - 2*trajectory[i] + trajectory[i-1]
                    curv = np.linalg.norm(second_deriv)
                    curvs.append(curv)
                return np.mean(curvs) if curvs else 0.0
            
            curv_z = estimate_curvature(reps_z)
            curv_h = estimate_curvature(reps_h)
            
            curvatures_z.append(curv_z)
            curvatures_h.append(curv_h)
    
    return np.array(curvatures_z), np.array(curvatures_h)


# =================================
# Main Execution
# =================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs_collapse', type=int, default=20)
    parser.add_argument('--num_epochs_guillotine', type=int, default=50)
    args = parser.parse_args()
    
    logger.info('='*60)
    logger.info('Starting Projection Head Geometry Experiments')
    logger.info('='*60)
    logger.info(f'Device: {device}')
    logger.info(f'PyTorch version: {torch.__version__}')
    
    # Create configuration
    config = ExperimentConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_epochs_collapse=args.num_epochs_collapse,
        num_epochs_guillotine=args.num_epochs_guillotine
    )
    DS = config.dataset
    
    # Log configuration
    logger.info('')
    logger.info('Configuration:')
    logger.info(f'  Dataset: {config.dataset}')
    logger.info(f'  Batch size: {config.batch_size}')
    logger.info(f'  Num seeds: {config.num_seeds}')
    logger.info(f'  Collapse epochs: {config.num_epochs_collapse}')
    logger.info(f'  Guillotine epochs: {config.num_epochs_guillotine}')
    logger.info(f'  Probe epochs: {config.exp2_probe_epochs}')
    logger.info(f'  Learning rates: collapse={config.lr_collapse}, guillotine={config.lr_guillotine}')
    logger.info('')
    
    # Create results directory structure
    results_dir = f'results/{DS}'
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f'Results directory: {results_dir}/')
    logger.info('')

    # Track total runtime
    start_time = time.time()
    
    # Experiment 1: Collapse Dynamics
    logger.info('='*60)
    logger.info(f'EXPERIMENT 1: Collapse Dynamics ({DS.upper()})')
    logger.info('='*60)
    
    results_exp1 = {}
    activations = ['linear', 'relu', 'gelu', 'swish']
    
    for activation in activations:
        try:
            results_exp1[activation] = run_collapse_experiment(DS, activation, config)
        except Exception as e:
            logger.info(f'Error with {activation}: {e}')
            continue
    
    # Save results
    np.save(f'{results_dir}/collapse_results.npy', results_exp1)
    logger.info(f'Saved collapse data to {results_dir}/collapse_results.npy')

    # Plotting Exp 1
    plt.figure(figsize=(10, 6.18))
    
    colors = {'linear': '#1f77b4', 'relu': '#ff7f0e', 'gelu': '#2ca02c', 'swish': '#d62728'}
    styles = {'linear': '--', 'relu': '-', 'gelu': ':', 'swish': '-.'}
    
    for activation in activations:
        if activation in results_exp1:
            epochs = range(len(results_exp1[activation]['mean']))
            plt.plot(epochs, results_exp1[activation]['mean'], 
                    label=f'{activation.upper()}', 
                    linestyle=styles.get(activation, '-'),
                    color=colors.get(activation),
                    linewidth=2)
            plt.fill_between(epochs, 
                           results_exp1[activation]['mean'] - results_exp1[activation]['std'],
                           results_exp1[activation]['mean'] + results_exp1[activation]['std'], 
                           alpha=0.2,
                           color=colors.get(activation))
    
    plt.title(f'Collapse Instability ({DS.upper()})', fontsize=12, fontweight='bold')
    plt.xlabel('Training Epochs', fontsize=11)
    plt.ylabel('Representation Std. Dev.', fontsize=11)
    plt.legend(frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig1_collapse_instability.png', dpi=300, bbox_inches='tight')
    logger.info(f'Saved plot to {results_dir}/fig1_collapse_instability.png')

    # Experiment 2a: Guillotine Effect
    logger.info('')
    logger.info('='*60)
    logger.info(f'EXPERIMENT 2a: Guillotine Effect ({DS.upper()})')
    logger.info('='*60)
    
    backbone_pretrained = None
    head_pretrained = None
    try:
        acc_z_linear, acc_h_linear, acc_z_mlp, acc_h_mlp, backbone_pretrained, head_pretrained = run_guillotine_experiment(DS, config)
        
        # Compute nonlinearity gaps
        gap_z = acc_z_mlp - acc_z_linear  # How much MLP helps on backbone
        gap_h = acc_h_mlp - acc_h_linear  # How much MLP helps on head
        
        # Save results
        np.save(f'{results_dir}/guillotine_results.npy', {
            'backbone_acc_linear': acc_z_linear,
            'head_acc_linear': acc_h_linear,
            'backbone_acc_mlp': acc_z_mlp,
            'head_acc_mlp': acc_h_mlp,
            'gap_backbone': gap_z,
            'gap_head': gap_h,
            'dataset': DS
        })
        logger.info(f'Saved guillotine data to {results_dir}/guillotine_results.npy')
 
        logger.info(f'Final Results ({DS.upper()}):')
        logger.info(f'  Backbone - Linear Probe: {acc_z_linear:.4f}')
        logger.info(f'  Backbone - MLP Probe:    {acc_z_mlp:.4f} (Gap: +{gap_z:.4f})')
        logger.info(f'  Head - Linear Probe:     {acc_h_linear:.4f}')
        logger.info(f'  Head - MLP Probe:        {acc_h_mlp:.4f} (Gap: +{gap_h:.4f})')
        logger.info(f'  Linear Loss (Guillotine): {acc_z_linear - acc_h_linear:.4f}')
        logger.info(f'  Nonlinearity Gap (Head vs Backbone): {gap_h - gap_z:.4f}') 
    except Exception as e:
        logger.info(f'Error in Guillotine experiment: {e}')
        backbone_pretrained = None
        head_pretrained = None
    
    # Experiment 2b: Manifold Curvature
    # Creates combined figure with probing (left) and curvature (right)
    if backbone_pretrained is not None and head_pretrained is not None:
        logger.info('')
        logger.info('='*60)
        logger.info(f'EXPERIMENT 2b: Manifold Curvature ({DS.upper()})')
        logger.info('='*60)
        logger.info(f'Running with {config.num_seeds} seeds for confidence intervals')
        
        try:
            all_curvs_z = []
            all_curvs_h = []
            all_ratios = []
            
            # Run curvature experiment with multiple random seeds
            for seed_idx in range(config.num_seeds):
                logger.info(f'  Seed {seed_idx + 1}/{config.num_seeds}')
                
                # Set seed for reproducible sampling
                set_seed(seed_idx)
                
                curvs_z, curvs_h = run_curvature_experiment(
                    DS, backbone_pretrained, head_pretrained, config, num_samples=200
                )
                
                mean_curv_z = np.mean(curvs_z)
                mean_curv_h = np.mean(curvs_h)
                ratio = mean_curv_h / (mean_curv_z + 1e-8)
                
                all_curvs_z.append(mean_curv_z)
                all_curvs_h.append(mean_curv_h)
                all_ratios.append(ratio)
                
                logger.info(f'    Backbone: {mean_curv_z:.6f}, Head: {mean_curv_h:.6f}, Ratio: {ratio:.2f}x')
            
            # Compute overall statistics
            mean_curv_z_overall = np.mean(all_curvs_z)
            std_curv_z_overall = np.std(all_curvs_z)
            mean_curv_h_overall = np.mean(all_curvs_h)
            std_curv_h_overall = np.std(all_curvs_h)
            mean_ratio = np.mean(all_ratios)
            std_ratio = np.std(all_ratios)
            
            logger.info(f'')
            logger.info(f'Final Results (Manifold Curvature - {config.num_seeds} seeds):')
            logger.info(f'  Backbone Curvature: {mean_curv_z_overall:.6f} ± {std_curv_z_overall:.6f}')
            logger.info(f'  Head Curvature:     {mean_curv_h_overall:.6f} ± {std_curv_h_overall:.6f}')
            logger.info(f'  Curvature Ratio:    {mean_ratio:.2f}x ± {std_ratio:.2f}x')
            
            # Save results
            np.save(f'{results_dir}/curvature_results.npy', {
                'curvatures_z_per_seed': all_curvs_z,
                'curvatures_h_per_seed': all_curvs_h,
                'ratios_per_seed': all_ratios,
                'mean_curv_z': mean_curv_z_overall,
                'std_curv_z': std_curv_z_overall,
                'mean_curv_h': mean_curv_h_overall,
                'std_curv_h': std_curv_h_overall,
                'mean_ratio': mean_ratio,
                'std_ratio': std_ratio,
                'num_seeds': config.num_seeds,
                'dataset': DS
            })
            logger.info(f'Saved curvature data to {results_dir}/curvature_results.npy')
            
            # Create combined figure: Probing (left) + Curvature (right)
            plt.figure(figsize=(14, 6.18))
            
            # Left subplot: Probing results (from Exp 2)
            plt.subplot(1, 2, 1)
            x_bar = np.arange(2)
            width = 0.35
            
            bars1 = plt.bar(x_bar - width/2, [acc_z_linear, acc_h_linear], width,
                           label='Linear Probe', color='#1f77b4', edgecolor='black', linewidth=1.5)
            bars2 = plt.bar(x_bar + width/2, [acc_z_mlp, acc_h_mlp], width,
                           label='MLP Probe', color='#ff7f0e', edgecolor='black', linewidth=1.5)
            
            plt.ylabel('Rotation Accuracy', fontsize=11)
            plt.title(f'Linear Probing Loss ({DS.upper()})', fontsize=12, fontweight='bold')
            plt.xticks(x_bar, ['Backbone (z)', 'Head (h(z))'])
            plt.ylim(0, 1.0)
            plt.axhline(0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            plt.legend(frameon=True, shadow=True)
            plt.grid(alpha=0.3, axis='y', linestyle=':')
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Right subplot: Curvature results  
            plt.subplot(1, 2, 2)
            x = np.arange(2)
            width = 0.5
            
            bars = plt.bar(x, [mean_curv_z_overall, mean_curv_h_overall], width,
                          yerr=[std_curv_z_overall, std_curv_h_overall],
                          color=['#1f77b4', '#d62728'],
                          edgecolor='black',
                          linewidth=1.5,
                          capsize=10,
                          error_kw={'linewidth': 2, 'ecolor': 'black'})
            
            plt.ylabel('Local Curvature', fontsize=11)
            plt.title(f'Manifold Curvature ({DS.upper()}, {config.num_seeds} seeds)',
                     fontsize=12, fontweight='bold')
            plt.xticks(x, ['Backbone (z)', 'Head (h(z))'])
            plt.grid(alpha=0.3, axis='y', linestyle=':')
            
            # Add value labels on bars with proper newlines
            for i, bar in enumerate(bars):
                height = bar.get_height()
                std_val = [std_curv_z_overall, std_curv_h_overall][i]
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}' + '\n' + f'±{std_val:.4f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add text box with curvature ratio
            textstr = f'Curvature Ratio:' + '\n' + f'{mean_ratio:.2f}x ± {std_ratio:.2f}x'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right', bbox=props,
                    fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{results_dir}/fig2_geometric_mechanisms.png', dpi=300, bbox_inches='tight')
            logger.info(f'Saved combined plot to {results_dir}/fig2_geometric_mechanisms.png')
        except Exception as e:
            logger.info(f'Error in Curvature experiment: {e}')
    else:
        logger.info('Skipping Experiment 2b: No pretrained models available')
    
    # Final summary
    total_time = time.time() - start_time
    logger.info('')
    logger.info('='*60)
    logger.info('EXPERIMENTS COMPLETED')
    logger.info('='*60)
    logger.info(f'Total runtime: {total_time/60:.2f} minutes ({total_time:.1f} seconds)')
    logger.info(f'Results saved to: {results_dir}/')
    logger.info(f'Log file: {log_file}')
    logger.info('='*60)
