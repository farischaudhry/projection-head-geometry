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
        'figure.figsize': (10, 7),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
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
        # Modified ResNet18 for CIFAR (no first maxpool)
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
        use_bn: Whether to use batch normalization
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
        logger.info(f"[{dataset_name}] Exp 1: {activation_type} head (Seed {seed})...")
        
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
# Experiment 2: Guillotine Effect
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


def run_guillotine_experiment(dataset_name: str, config: ExperimentConfig) -> tuple[float, float]:
    """
    Test information loss across backbone-head boundary.
    
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
    
    logger.info(f'[{dataset_name}] Exp 2: Pre-training SimCLR ({config.num_epochs_guillotine} Epochs)')
    
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
    
    # Train probes
    probe_z = nn.Linear(512, 4).to(device)
    probe_h = nn.Linear(2048, 4).to(device)
    opt_z = optim.Adam(probe_z.parameters(), lr=1e-3)
    opt_h = optim.Adam(probe_h.parameters(), lr=1e-3)
    
    for epoch in tqdm(range(config.exp2_probe_epochs), desc='Training Probes', leave=False):
        for img, rot_label in rot_loader:
            img, rot_label = img.to(device), rot_label.to(device)
            
            with torch.no_grad():
                feat_z = backbone(img)
                feat_h = head(feat_z)
            
            # Train backbone probe
            loss_z = F.cross_entropy(probe_z(feat_z), rot_label)
            opt_z.zero_grad()
            loss_z.backward()
            opt_z.step()
            
            # Train head probe
            loss_h = F.cross_entropy(probe_h(feat_h), rot_label)
            opt_h.zero_grad()
            loss_h.backward()
            opt_h.step()

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

    return evaluate(probe_z, False), evaluate(probe_h, True)


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
    
    logger.info(f'using device: {device}')

    # Create configuration
    config = ExperimentConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_epochs_collapse=args.num_epochs_collapse,
        num_epochs_guillotine=args.num_epochs_guillotine
    )
    DS = config.dataset

    # Experiment 1: Collapse Dynamics
    logger.info(f'Starting Experiment 1: Collapse Dynamics ({DS})')
    
    results_exp1 = {}
    activations = ['linear', 'relu', 'gelu', 'swish']
    
    for activation in activations:
        try:
            results_exp1[activation] = run_collapse_experiment(DS, activation, config)
        except Exception as e:
            logger.info(f'Error with {activation}: {e}')
            continue
    
    # Save results
    np.save(f'collapse_results_{DS}.npy', results_exp1)
    logger.info(f'Saved collapse data to collapse_results_{DS}.npy')

    # Plotting Exp 1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    colors = {'linear': '#1f77b4', 'relu': '#ff7f0e', 'gelu': '#2ca02c', 'swish': '#d62728'}
    styles = {'linear': '--', 'relu': '-', 'gelu': '-', 'swish': '-.'}
    
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
    plt.ylabel('Representation Std Dev', fontsize=11)
    plt.legend(frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle=':')

    # Experiment 2: Guillotine Effect
    logger.info(f'Starting Experiment 2: Guillotine Effect ({DS})')
    
    try:
        acc_z, acc_h = run_guillotine_experiment(DS, config)
        
        # Save results
        np.save(f'guillotine_results_{DS}.npy', {
            'backbone_acc': acc_z, 
            'head_acc': acc_h,
            'dataset': DS
        })
        logger.info(f'Saved guillotine data to guillotine_results_{DS}.npy')
        
        # Plotting Exp 2
        plt.subplot(1, 2, 2)
        bars = plt.bar(['Backbone (z)', 'Head (h(z))'], [acc_z, acc_h], 
                      color=['#1f77b4', '#d62728'], 
                      edgecolor='black', 
                      linewidth=1.5)
        plt.ylabel('Rotation Accuracy', fontsize=11)
        plt.title(f'Information Loss ({DS.upper()})', fontsize=12, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.axhline(0.25, color='gray', linestyle='--', label='Random Chance', linewidth=1)
        plt.legend()
        plt.grid(alpha=0.3, axis='y', linestyle=':')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        logger.info(f'Final Results ({DS.upper()}):')
        logger.info(f'  Backbone Accuracy: {acc_z:.4f}')
        logger.info(f'  Head Accuracy:     {acc_h:.4f}')
        logger.info(f'  Information Loss:  {acc_z - acc_h:.4f}')
        
    except Exception as e:
        logger.info(f'Error in Guillotine experiment: {e}')
    
    plt.tight_layout()
    plt.savefig(f'fig1_geometric_mechanisms_{DS}.png', dpi=300, bbox_inches='tight')
    logger.info(f'Saved plot to fig1_geometric_mechanisms_{DS}.png')
