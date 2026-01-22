import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
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


class VisionTransformerBackbone(nn.Module):
    """Vision Transformer (ViT-Tiny) backbone adapted for CIFAR."""
    def __init__(self, output_dim=192):
        super().__init__()
        # ViT-Tiny configuration: patch_size=4 for 32x32 images (CIFAR)
        # This gives 8x8 = 64 patches
        self.output_dim = output_dim
        self.patch_size = 4
        self.num_patches = (32 // self.patch_size) ** 2  # 64 patches
        self.embed_dim = 192  # ViT-Tiny dimension
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, 
                                     kernel_size=self.patch_size, 
                                     stride=self.patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Transformer encoder (3 layers for ViT-Tiny)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=3,  # ViT-Tiny uses 3 heads
            dim_feedforward=768,  # 4x embed_dim
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: (B, 3, 32, 32) -> (B, embed_dim, 8, 8) -> (B, 64, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Return cls token representation
        return x[:, 0]  # (B, embed_dim)


def get_backbone(architecture='resnet18'):
    """Factory function to get backbone by name."""
    if architecture == 'resnet18':
        return ResNetBackbone(output_dim=512)
    elif architecture == 'vit_tiny':
        return VisionTransformerBackbone(output_dim=192)
    else:
        raise ValueError(f'Unknown architecture: {architecture}')


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
    seeds: list[int] = None,
    architecture: str = 'resnet18'
) -> dict:
    """
    Test collapse instability with different activation functions.
    Initialize projector head to pseudo-collapsed state and track variance of representations.
    
    Args:
        dataset_name: Name of dataset ('cifar10' or 'cifar100')
        activation_type: Type of activation ('linear', 'relu', 'gelu', 'swish')
        config: Experiment configuration
        seeds: List of random seeds
        architecture: Backbone architecture ('resnet18' or 'vit_tiny')
        
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
        backbone = get_backbone(architecture).to(device)
        projector = ProjectionHead(
            input_dim=backbone.output_dim,
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
        
        # # Verify collapse initialization
        # with torch.no_grad():
        #     sample_batch = next(iter(train_loader))
        #     x1, x2 = sample_batch[0][0].to(device), sample_batch[0][1].to(device)
        #     z1 = projector(backbone(x1))
        #     z1_normalized = F.normalize(z1, dim=1)
        #     initial_var = torch.var(z1_normalized, dim=0).mean().item()
        #     if seed == 0:  # Log only for first seed to avoid clutter
        #         logger.info(f'[{dataset_name}] Pseudo-collapsed init variance: {initial_var:.6f}')
        #         if initial_var > 1e-2:
        #             logger.warning(f'WARNING: Initial variance {initial_var:.6f} may be too high for collapse dynamics')
        
        variance_history = []
        logger.info(f'[{dataset_name}] Exp 1: {activation_type} head (Seed {seed})')
        
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
# Experiment 2a: Guillotine Effect
# Experiment 2b: Manifold Curvature
# Experiment 2c: Orbit Visualization
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


def run_orbit_visualization_experiment(
    dataset_name: str,
    backbone: nn.Module,
    head: nn.Module,
    config: ExperimentConfig,
    num_classes: int = 5,
    images_per_class: int = 3,
    num_rotations: int = 12
) -> dict:
    """
    Collect augmentation orbit data for visualization of metric singularity.
    
    For each class, we sample a few images and apply multiple rotations,
    collecting both backbone and head representations. This visualizes how
    the projection head collapses augmentation orbits (Proposition 5.1).
    
    Args:
        dataset_name: Name of dataset
        backbone: Pretrained backbone model
        head: Pretrained projection head
        config: Experiment configuration
        num_classes: Number of CIFAR classes to visualize
        images_per_class: Number of images per class
        num_rotations: Number of rotation steps (0-360 degrees)
        
    Returns:
        Dictionary containing orbit data for backbone and head
    """
    backbone.eval()
    head.eval()
    
    # Load test dataset without transforms
    ds = get_dataset(dataset_name, train=False, transform=transforms.ToTensor())
    
    # Select specific classes (evenly distributed)
    if dataset_name == 'cifar10':
        all_classes = list(range(10))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        all_classes = list(range(100))
        class_names = [f'Class_{i}' for i in range(100)]
    
    # Select evenly spaced classes
    selected_classes = [all_classes[i] for i in np.linspace(0, len(all_classes)-1, num_classes, dtype=int)]
    
    logger.info(f'[{dataset_name}] Orbit Visualization: Collecting data for {num_classes} classes')
    logger.info(f'  Selected classes: {[class_names[c] for c in selected_classes]}')
    logger.info(f'  Images per class: {images_per_class}')
    logger.info(f'  Rotations per image: {num_rotations} (0-360°)')
    
    # Rotation angles
    angles = np.linspace(0, 360, num_rotations, endpoint=False)
    
    # Storage for orbit data
    orbits_z = []  # Backbone representations
    orbits_h = []  # Head representations
    orbit_labels = []  # Class labels
    orbit_ids = []  # Unique orbit ID for each image
    orbit_angles = []  # Rotation angles
    
    # Collect data for each class
    orbit_id = 0
    with torch.no_grad():
        for class_idx in tqdm(selected_classes, desc='Collecting Orbits'):
            # Find images for this class
            class_images = [(img, lbl) for img, lbl in ds if lbl == class_idx]
            
            # Sample random images
            sampled_images = random.sample(class_images, min(images_per_class, len(class_images)))
            
            for img, lbl in sampled_images:
                img_tensor = img.unsqueeze(0).to(device)
                
                # Collect representations at different rotations
                for angle in angles:
                    # Apply rotation
                    img_rot = transforms.functional.rotate(img_tensor, float(angle))
                    
                    # Get representations
                    z = F.normalize(backbone(img_rot), dim=1)
                    h = F.normalize(head(z), dim=1)
                    
                    # Store
                    orbits_z.append(z.cpu().numpy().flatten())
                    orbits_h.append(h.cpu().numpy().flatten())
                    orbit_labels.append(class_idx)
                    orbit_ids.append(orbit_id)
                    orbit_angles.append(angle)
                
                orbit_id += 1
    
    # Convert to numpy arrays
    orbits_z = np.array(orbits_z)
    orbits_h = np.array(orbits_h)
    orbit_labels = np.array(orbit_labels)
    orbit_ids = np.array(orbit_ids)
    orbit_angles = np.array(orbit_angles)
    
    logger.info(f'  Collected {len(orbits_z)} total representations')
    logger.info(f'  Backbone shape: {orbits_z.shape}')
    logger.info(f'  Head shape: {orbits_h.shape}')
    
    return {
        'orbits_z': orbits_z,
        'orbits_h': orbits_h,
        'orbit_labels': orbit_labels,
        'orbit_ids': orbit_ids,
        'orbit_angles': orbit_angles,
        'selected_classes': selected_classes,
        'class_names': [class_names[c] for c in selected_classes],
        'num_rotations': num_rotations,
        'num_classes': num_classes,
        'images_per_class': images_per_class,
        'dataset': dataset_name
    }


# =================================
# Main Execution
# =================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs_collapse', type=int, default=20)
    parser.add_argument('--num_epochs_guillotine', type=int, default=50)
    parser.add_argument('--architecture', type=str, default='resnet18', 
                        choices=['resnet18', 'vit_tiny'],
                        help='Backbone architecture for collapse experiment')
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
    ARCH = args.architecture
    
    # Log configuration
    logger.info('Configuration:')
    logger.info(f'  Dataset: {config.dataset}')
    logger.info(f'  Batch size: {config.batch_size}')
    logger.info(f'  Num seeds: {config.num_seeds}')
    logger.info(f'  Collapse epochs: {config.num_epochs_collapse}')
    logger.info(f'  Guillotine epochs: {config.num_epochs_guillotine}')
    logger.info(f'  Probe epochs: {config.exp2_probe_epochs}')
    logger.info(f'  Learning rates: collapse={config.lr_collapse}, guillotine={config.lr_guillotine}')
    
    # Create results directory structure
    results_dir = f'results/{DS}/{ARCH}'
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f'Results directory: {results_dir}/')

    # Track total runtime
    start_time = time.time()
    
    # Experiment 1: Collapse Dynamics
    logger.info('='*60)
    logger.info(f'EXPERIMENT 1: Collapse Dynamics ({DS.upper()}, {ARCH})')
    logger.info('='*60)
    
    results_exp1 = {}
    activations = ['linear', 'relu', 'gelu', 'swish']
    for activation in activations:
        try:
            results_exp1[activation] = run_collapse_experiment(DS, activation, config, architecture=ARCH)
        except Exception as e:
            logger.info(f'Error with {activation}: {e}')
            continue
    
    # Save results
    np.save(f'{results_dir}/collapse_results.npy', results_exp1)
    logger.info(f'Saved collapse data to {results_dir}/collapse_results.npy')

    # Experiment 2a: Guillotine Effect
    logger.info('='*60)
    logger.info(f'EXPERIMENT 2a: Guillotine Effect ({DS.upper()}, {ARCH})')
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
    if backbone_pretrained is not None and head_pretrained is not None:
        logger.info('='*60)
        logger.info(f'EXPERIMENT 2b: Manifold Curvature ({DS.upper()}, {ARCH})')
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
        except Exception as e:
            logger.info(f'Error in Curvature experiment: {e}')
    else:
        logger.info('Skipping Experiment 2b: No pretrained models available')
    
    # Experiment 2c: Orbit Visualization
    if backbone_pretrained is not None and head_pretrained is not None:
        logger.info('='*60)
        logger.info(f'EXPERIMENT 2c: Orbit Visualization ({DS.upper()}, {ARCH})')
        logger.info('='*60)
        
        try:
            orbit_data = run_orbit_visualization_experiment(
                DS, backbone_pretrained, head_pretrained, config,
                num_classes=5, images_per_class=3, num_rotations=12
            )
            
            # Save orbit data
            np.save(f'{results_dir}/orbit_visualization.npy', orbit_data)
            logger.info(f'Saved orbit data to {results_dir}/orbit_visualization.npy')
            logger.info(f'Total orbits: {orbit_data["orbit_ids"].max() + 1}')
            logger.info(f'Total points: {len(orbit_data["orbits_z"])}')
        except Exception as e:
            logger.info(f'Error in Orbit Visualization experiment: {e}')
    else:
        logger.info('Skipping Experiment 2c: No pretrained models available')
    
    # Final summary
    total_time = time.time() - start_time
    logger.info('='*60)
    logger.info('EXPERIMENTS COMPLETED')
    logger.info('='*60)
    logger.info(f'Total runtime: {total_time/60:.2f} minutes ({total_time:.1f} seconds)')
    logger.info(f'Results saved to: {results_dir}/')
    logger.info(f'Log file: {log_file}')
    logger.info('To generate figures, run:')
    logger.info(f'  python plot_results.py --dataset {DS} --architecture {ARCH}')
    logger.info('='*60)
