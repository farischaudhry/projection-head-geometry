import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from tqdm import tqdm
import os
import random
import argparse
import logging

from experiments import (
    get_backbone, 
    TwoCropTransform,
    get_dataset,
    ExperimentConfig,
    run_curvature_experiment,
    run_orbit_visualization_experiment
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# FLEXIBLE HEAD (For Depth Ablations)
class FlexibleProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048, num_layers=2, activation='swish', use_bn=True):
        super().__init__()
        layers = []
        if activation == 'relu': act_fn = nn.ReLU
        elif activation == 'gelu': act_fn = nn.GELU
        elif activation == 'swish': act_fn = nn.SiLU
        else: act_fn = nn.Identity

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm1d(output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=not use_bn))
                if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_fn())
            layers.append(nn.Linear(hidden_dim, output_dim, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm1d(output_dim))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# DEPTH ABLATIONS
class RotationDataset(Dataset):
    def __init__(self, dataset_name, train=True, seed=None):
        self.ds = get_dataset(dataset_name, train=train, transform=None)
        self.to_tensor = transforms.ToTensor()
        rng = random.Random(seed) if seed else random
        self.rotations = [rng.randint(0, 3) for _ in range(len(self.ds))]
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        rot_idx = self.rotations[idx]
        return self.to_tensor(transforms.functional.rotate(img, rot_idx * 90)), rot_idx

def run_geometry_ablation(args):
    logger.info(f"========== STARTING GEOMETRIC ABLATION: DEPTH {args.num_layers} ==========")
    config = ExperimentConfig()
    set_seed(0)
    
    # Pretrain SimCLR with specific depth
    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45), transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ])
    train_ds = get_dataset(config.dataset, train=True, transform=TwoCropTransform(simclr_transform))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
    
    backbone = get_backbone('resnet18').to(device)
    head = FlexibleProjectionHead(input_dim=backbone.output_dim, num_layers=args.num_layers, activation='swish', use_bn=True).to(device)
    optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=config.lr_guillotine)
    
    for epoch in tqdm(range(config.num_epochs_guillotine), desc=f'SimCLR Pre-training (Depth {args.num_layers})'): 
        backbone.train(); head.train()
        for (x1, x2), _ in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = F.normalize(head(backbone(x1)), dim=1), F.normalize(head(backbone(x2)), dim=1)
            loss = F.cross_entropy(torch.mm(z1, z2.t()) / 0.1, torch.arange(x1.size(0)).to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Guillotine Probes
    backbone.eval(); head.eval()
    rot_loader = DataLoader(RotationDataset(config.dataset, train=True, seed=0), batch_size=256, shuffle=True)
    test_rot_loader = DataLoader(RotationDataset(config.dataset, train=False, seed=0), batch_size=256, shuffle=False)
    
    probe_z = nn.Linear(backbone.output_dim, 4).to(device)
    probe_h = nn.Linear(2048, 4).to(device)
    opt_z, opt_h = optim.Adam(probe_z.parameters(), lr=1e-3), optim.Adam(probe_h.parameters(), lr=1e-3)
    
    for _ in tqdm(range(config.exp2_probe_epochs), desc='Training Probes', leave=False):
        for img, rot in rot_loader:
            img, rot = img.to(device), rot.to(device)
            with torch.no_grad(): feat_z, feat_h = backbone(img), head(backbone(img))
            loss_z = F.cross_entropy(probe_z(feat_z), rot)
            opt_z.zero_grad(); loss_z.backward(); opt_z.step()
            loss_h = F.cross_entropy(probe_h(feat_h), rot)
            opt_h.zero_grad(); loss_h.backward(); opt_h.step()

    def evaluate(probe, use_head):
        corr, tot = 0, 0
        with torch.no_grad():
            for img, rot in test_rot_loader:
                img, rot = img.to(device), rot.to(device)
                preds = probe(head(backbone(img)) if use_head else backbone(img)).argmax(dim=1)
                corr += (preds == rot).sum().item()
                tot += rot.size(0)
        return corr / tot
    
    acc_z, acc_h = evaluate(probe_z, False), evaluate(probe_h, True)
    logger.info(f"[Guillotine] Depth {args.num_layers} | Backbone: {acc_z:.4f} | Head: {acc_h:.4f}")
    
    # Curvature & Orbits 
    curv_z, curv_h = run_curvature_experiment(config.dataset, backbone, head, config)
    orbit_data = run_orbit_visualization_experiment(config.dataset, backbone, head, config)
    
    output_file = f'geometric_ablation_depth_{args.num_layers}.npz'
    np.savez(output_file, depth=args.num_layers, acc_z_linear=acc_z, acc_h_linear=acc_h,
             curvatures_z=curv_z, curvatures_h=curv_h, **{k:v for k,v in orbit_data.items() if isinstance(v, np.ndarray)})
    logger.info(f"Saved Geometry Ablation to {output_file}\n")

# PLOTTING & METRICS
def _calculate_geometric_metrics(X, ids, labels):
    """High-dimensional geometric metrics calculation helper."""
    intra_distances = [np.mean(pdist(X[ids == oid], metric='euclidean')) for oid in np.unique(ids) if len(X[ids == oid]) > 1]
    d_intra_mean, d_intra_std = np.mean(intra_distances), np.std(intra_distances)

    class_centroids = np.array([np.mean(X[labels == cls], axis=0) for cls in np.unique(labels)])
    inter_dists = pdist(class_centroids, metric='euclidean')
    d_inter_mean, d_inter_std = np.mean(inter_dists), np.std(inter_dists)  
    
    variances = [np.mean(np.sum((X[ids == oid] - np.mean(X[ids == oid], axis=0))**2, axis=1)) for oid in np.unique(ids)]
    spread_mean, spread_std = np.mean(variances), np.std(variances)
    
    ratio = d_inter_mean / (d_intra_mean + 1e-10)
    ratio_std = ratio * np.sqrt((d_inter_std / (d_inter_mean + 1e-10))**2 + (d_intra_std / (d_intra_mean + 1e-10))**2)
    
    return {
        'd_intra': (d_intra_mean, d_intra_std), 'd_inter': (d_inter_mean, d_inter_std),
        'spread': (spread_mean, spread_std), 'ratio': (ratio, ratio_std)
    }

def plot_all_depth_orbits():
    """Loops through all depth ablations and runs the geometric PCA and metrics."""
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for depth in [1, 2, 3]:
        file_path = f'geometric_ablation_depth_{depth}.npz'
        if not os.path.exists(file_path):
            print(f"Skipping Depth {depth}: {file_path} not found.")
            continue
            
        print(f"\nProcessing Geometry Ablation for Depth {depth}...")
        data = np.load(file_path)
        orbits_z, orbits_h = data['orbits_z'], data['orbits_h']
        orbit_labels, orbit_ids = data['orbit_labels'], data['orbit_ids']
        
        selected_classes = np.unique(orbit_labels)
        class_names = [cifar10_classes[c] for c in selected_classes]
        
        metrics_z = _calculate_geometric_metrics(orbits_z, orbit_ids, orbit_labels)
        metrics_h = _calculate_geometric_metrics(orbits_h, orbit_ids, orbit_labels)

        mean_z, std_z = metrics_z['spread']
        mean_h, std_h = metrics_h['spread']
        compression_ratio = mean_z / (mean_h + 1e-10)

        # Print master table
        print('\n' + '='*60)
        print(f'High-dimensional Metrics (Depth {depth} Head)')
        print('='*60)
        print(f"{'Metric':<25} | {'Backbone (z)':<18} | {'Head (h(z))':<18}")
        print('-'*60)
        print(f"{'Orbit Spread':<25} | {mean_z:.8f} ± {std_z:.8f} | {mean_h:.8f} ± {std_h:.8f}")
        print(f"{'Intra-orbit (D_intra)':<25} | {metrics_z['d_intra'][0]:.4f} ± {metrics_z['d_intra'][1]:.4f} | {metrics_h['d_intra'][0]:.4f} ± {metrics_h['d_intra'][1]:.4f}")
        print(f"{'Inter-class (D_inter)':<25} | {metrics_z['d_inter'][0]:.4f} ± {metrics_z['d_inter'][1]:.4f} | {metrics_h['d_inter'][0]:.4f} ± {metrics_h['d_inter'][1]:.4f}")
        print(f"{'Class/Orbit Ratio':<25} | {metrics_z['ratio'][0]:.2f}x ± {metrics_z['ratio'][1]:.2f}x | {metrics_h['ratio'][0]:.2f}x ± {metrics_h['ratio'][1]:.2f}x")
        print('='*60 + '\n')
        print(f'Orbit Compression: {compression_ratio:.2f}x')
        print(f'Separation Improvement: {metrics_z["ratio"][0]:.2f}x → {metrics_h["ratio"][0]:.2f}x ({metrics_h["ratio"][0] / metrics_z["ratio"][0]:.2f}x better)\n')
        
        # PCA Projection
        orbits_z_norm = orbits_z / (np.linalg.norm(orbits_z, axis=1, keepdims=True) + 1e-8)
        orbits_h_norm = orbits_h / (np.linalg.norm(orbits_h, axis=1, keepdims=True) + 1e-8)
        z_2d = PCA(n_components=2, random_state=42).fit_transform(orbits_z_norm)
        h_2d = PCA(n_components=2, random_state=42).fit_transform(orbits_h_norm)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        class_colors = {label: colors[i] for i, label in enumerate(selected_classes)}
        
        # Backbone Space
        ax = axes[0]
        for orbit_id in np.unique(orbit_ids):
            mask = orbit_ids == orbit_id
            class_label = orbit_labels[mask][0]
            ax.plot(z_2d[mask, 0], z_2d[mask, 1], color=class_colors[class_label], alpha=0.6, linewidth=1.5, zorder=1)
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], color=class_colors[class_label], s=30, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=2)
            ax.scatter(z_2d[mask][0:1, 0], z_2d[mask][0:1, 1], color=class_colors[class_label], s=120, marker='*', edgecolors='black', linewidths=1.5, zorder=3)
        ax.set_title('Backbone Representation Space $z$', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.text(0.05, 0.95, f'Mean Orbit Spread:\n{mean_z:.5f} ± {std_z:.5f}', transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontweight='bold')
        
        # Head Space
        ax = axes[1]
        marker_scale = min(5.0, np.sqrt(compression_ratio)) 
        for orbit_id in np.unique(orbit_ids):
            mask = orbit_ids == orbit_id
            class_label = orbit_labels[mask][0]
            ax.plot(h_2d[mask, 0], h_2d[mask, 1], color=class_colors[class_label], alpha=0.7, linewidth=2.5, zorder=1)
            ax.scatter(h_2d[mask, 0], h_2d[mask, 1], color=class_colors[class_label], s=30*marker_scale, alpha=0.8, edgecolors='white', linewidths=1.0, zorder=2)
            ax.scatter(h_2d[mask][0:1, 0], h_2d[mask][0:1, 1], color=class_colors[class_label], s=150*marker_scale, marker='*', edgecolors='black', linewidths=2.0, zorder=3)
        ax.set_title(f'Projection Head Space $h(z)$ (Depth {depth})', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.text(0.05, 0.95, f'Mean Orbit Spread:\n{mean_h:.5f} ± {std_h:.5f}\n({compression_ratio:.1f}× smaller)', transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontweight='bold')
        
        # Formatting
        legend_elements = [Patch(facecolor=class_colors[label], label=class_names[i]) for i, label in enumerate(selected_classes)]
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, markeredgecolor='black', label='Start (0°)', linestyle='None'))
        fig.legend(handles=legend_elements, loc='lower center', ncol=6, frameon=True, shadow=True, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        plt.suptitle(f'Visualization of Augmentation Orbit Collapse (Depth {depth} Head)', fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_name = f'fig3_orbit_visualization_depth_{depth}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['geometry', 'plot'])
    parser.add_argument('--num_layers', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    if args.task == 'geometry':
        run_geometry_ablation(args)
    elif args.task == 'plot':
        plot_all_depth_orbits()
