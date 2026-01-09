"""
Plot results from saved experiment data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
import os
from sklearn.decomposition import PCA

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


def plot_collapse_instability(results_dir, dataset, architecture):
    """Plot Experiment 1: Collapse Instability."""
    # Load results
    results = np.load(f'{results_dir}/collapse_results.npy', allow_pickle=True).item()
    
    plt.figure(figsize=(10, 6.18))
    
    colors = {'linear': '#1f77b4', 'relu': '#ff7f0e', 'gelu': '#2ca02c', 'swish': '#d62728'}
    styles = {'linear': '--', 'relu': '-', 'gelu': ':', 'swish': '-.'}
    activations = ['linear', 'relu', 'gelu', 'swish']

    for activation in activations:
        if activation in results:
            epochs = range(len(results[activation]['mean']))
            plt.plot(epochs, results[activation]['mean'], 
                    label=f'{activation.upper()}', 
                    linestyle=styles.get(activation, '-'),
                    color=colors.get(activation),
                    linewidth=2)
            plt.fill_between(epochs, 
                           results[activation]['mean'] - results[activation]['std'],
                           results[activation]['mean'] + results[activation]['std'], 
                           alpha=0.2,
                           color=colors.get(activation))
    
    plt.title(f'Collapse Instability ({dataset.upper()}, {architecture})', fontsize=12, fontweight='bold')
    plt.xlabel('Training Epochs', fontsize=11)
    plt.ylabel('Representation Std. Dev.', fontsize=11)
    
    # Force integer x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.legend(frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig1_collapse_instability.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig1_collapse_instability.png')


def plot_geometric_mechanisms(results_dir, dataset, architecture):
    """Plot Experiment 2b: Combined Guillotine + Curvature."""
    # Load results
    guillotine = np.load(f'{results_dir}/guillotine_results.npy', allow_pickle=True).item()
    curvature = np.load(f'{results_dir}/curvature_results.npy', allow_pickle=True).item()
    
    # Extract values
    acc_z_linear = guillotine['backbone_acc_linear']
    acc_h_linear = guillotine['head_acc_linear']
    acc_z_mlp = guillotine['backbone_acc_mlp']
    acc_h_mlp = guillotine['head_acc_mlp']
    
    mean_curv_z = curvature['mean_curv_z']
    std_curv_z = curvature['std_curv_z']
    mean_curv_h = curvature['mean_curv_h']
    std_curv_h = curvature['std_curv_h']
    mean_ratio = curvature['mean_ratio']
    std_ratio = curvature['std_ratio']
    num_seeds = curvature['num_seeds']
    
    # Create combined figure
    plt.figure(figsize=(14, 6.18))
    
    # Left subplot: Probing results
    plt.subplot(1, 2, 1)
    x_bar = np.arange(2)
    width = 0.35
    
    bars1 = plt.bar(x_bar - width/2, [acc_z_linear, acc_h_linear], width,
                   label='Linear Probe', color='#1f77b4', edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x_bar + width/2, [acc_z_mlp, acc_h_mlp], width,
                   label='MLP Probe', color='#ff7f0e', edgecolor='black', linewidth=1.5)
    
    plt.ylabel('Rotation Accuracy', fontsize=11)
    plt.title(f'Probing Loss ({dataset.upper()}, {architecture})', fontsize=12, fontweight='bold')
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
    
    bars = plt.bar(x, [mean_curv_z, mean_curv_h], width,
                  yerr=[std_curv_z, std_curv_h],
                  color=['#1f77b4', '#d62728'],
                  edgecolor='black',
                  linewidth=1.5,
                  capsize=10,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    plt.ylabel('Local Curvature', fontsize=11)
    plt.title(f'Manifold Curvature ({dataset.upper()}, {architecture}, {num_seeds} seeds)',
             fontsize=12, fontweight='bold')
    plt.xticks(x, ['Backbone (z)', 'Head (h(z))'])
    plt.grid(alpha=0.3, axis='y', linestyle=':')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        std_val = [std_curv_z, std_curv_h][i]
        y_pos = height + std_val * 1.15
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.4f} ± {std_val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    textstr = f'Curvature Ratio:\n{mean_ratio:.2f}x ± {std_ratio:.2f}x'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', bbox=props,
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig2_geometric_mechanisms.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig2_geometric_mechanisms.png')


def plot_orbit_visualization(results_dir, dataset, architecture):
    """Plot Experiment 2c: Orbit Visualization via PCA."""
    # Load orbit data
    orbit_data = np.load(f'{results_dir}/orbit_visualization.npy', allow_pickle=True).item()
    
    orbits_z = orbit_data['orbits_z']
    orbits_h = orbit_data['orbits_h']
    orbit_labels = orbit_data['orbit_labels']
    orbit_ids = orbit_data['orbit_ids']
    class_names = orbit_data['class_names']
    num_rotations = orbit_data['num_rotations']
    
    # DIAGNOSTIC: Compute orbit variances in high-dimensional space
    print('\n=== ORBIT VARIANCE DIAGNOSTICS (High-Dimensional Space) ===')
    orbit_variances_z = []
    orbit_variances_h = []
    
    for orbit_id in np.unique(orbit_ids):
        mask = orbit_ids == orbit_id
        
        # Compute variance of orbit points
        orbit_z = orbits_z[mask]
        orbit_h = orbits_h[mask]
        
        # Variance = mean squared distance from centroid
        centroid_z = orbit_z.mean(axis=0)
        centroid_h = orbit_h.mean(axis=0)
        
        var_z = np.mean(np.sum((orbit_z - centroid_z)**2, axis=1))
        var_h = np.mean(np.sum((orbit_h - centroid_h)**2, axis=1))
        
        orbit_variances_z.append(var_z)
        orbit_variances_h.append(var_h)
    
    mean_var_z = np.mean(orbit_variances_z)
    mean_var_h = np.mean(orbit_variances_h)
    
    print(f'Backbone (z) - Mean orbit variance: {mean_var_z:.6f}')
    print(f'Head (h(z))  - Mean orbit variance: {mean_var_h:.6f}')
    print(f'Orbit compression ratio: {mean_var_z / (mean_var_h + 1e-10):.2f}x')
    
    # Apply PCA to reduce to 2D for visualization
    print('\n=== PCA PROJECTION (L2-Normalized Representations) ===')
    # Normalize to unit sphere before PCA to preserve structure
    orbits_z_norm = orbits_z / (np.linalg.norm(orbits_z, axis=1, keepdims=True) + 1e-8)
    orbits_h_norm = orbits_h / (np.linalg.norm(orbits_h, axis=1, keepdims=True) + 1e-8)
    
    pca_z = PCA(n_components=2, random_state=42)
    z_2d = pca_z.fit_transform(orbits_z_norm)
    
    pca_h = PCA(n_components=2, random_state=42)
    h_2d = pca_h.fit_transform(orbits_h_norm)
    
    print(f'  Backbone PCA explained variance: {pca_z.explained_variance_ratio_.sum():.3f}')
    print(f'  Head PCA explained variance: {pca_h.explained_variance_ratio_.sum():.3f}')
    
    # DIAGNOSTIC: Compute orbit variances in 2D space
    print('\n=== ORBIT VARIANCE DIAGNOSTICS (2D PCA Space) ===')
    orbit_variances_z_2d = []
    orbit_variances_h_2d = []
    
    for orbit_id in np.unique(orbit_ids):
        mask = orbit_ids == orbit_id
        
        orbit_z_2d = z_2d[mask]
        orbit_h_2d = h_2d[mask]
        
        centroid_z_2d = orbit_z_2d.mean(axis=0)
        centroid_h_2d = orbit_h_2d.mean(axis=0)
        
        var_z_2d = np.mean(np.sum((orbit_z_2d - centroid_z_2d)**2, axis=1))
        var_h_2d = np.mean(np.sum((orbit_h_2d - centroid_h_2d)**2, axis=1))
        
        orbit_variances_z_2d.append(var_z_2d)
        orbit_variances_h_2d.append(var_h_2d)
    
    mean_var_z_2d = np.mean(orbit_variances_z_2d)
    mean_var_h_2d = np.mean(orbit_variances_h_2d)
    
    print(f'Backbone (z) - Mean 2D orbit variance: {mean_var_z_2d:.6f}')
    print(f'Head (h(z))  - Mean 2D orbit variance: {mean_var_h_2d:.6f}')
    print(f'2D Orbit compression ratio: {mean_var_z_2d / (mean_var_h_2d + 1e-10):.2f}x')
    print('='*60 + '\n')
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Calculate compression ratio for use in plot
    compression_ratio = mean_var_z / (mean_var_h + 1e-10)
    
    # Color palette for classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    class_colors = {label: colors[i] for i, label in enumerate(orbit_data['selected_classes'])}
    
    # Plot backbone space (left)
    ax = axes[0]
    for orbit_id in np.unique(orbit_ids):
        mask = orbit_ids == orbit_id
        class_label = orbit_labels[mask][0]
        
        # Plot orbit trajectory
        ax.plot(z_2d[mask, 0], z_2d[mask, 1], 
               color=class_colors[class_label], alpha=0.6, linewidth=1.5, zorder=1)
        
        # Plot points
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], 
                  color=class_colors[class_label], s=30, alpha=0.8, 
                  edgecolors='white', linewidths=0.5, zorder=2)
        
        # Mark start point
        ax.scatter(z_2d[mask][0:1, 0], z_2d[mask][0:1, 1],
                  color=class_colors[class_label], s=120, marker='*',
                  edgecolors='black', linewidths=1.5, zorder=3)
    
    ax.set_title('Backbone Representation Space $z$', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('PC 1', fontsize=12)
    ax.set_ylabel('PC 2', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    textstr = f'Mean Orbit Spread:\n{mean_var_z:.5f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Plot projection head space (right)
    ax = axes[1]
    
    # Use larger markers for head since orbits are smaller
    marker_scale = min(5.0, np.sqrt(compression_ratio)) 
    
    for orbit_id in np.unique(orbit_ids):
        mask = orbit_ids == orbit_id
        class_label = orbit_labels[mask][0]
        
        # Plot orbit trajectory (thicker)
        ax.plot(h_2d[mask, 0], h_2d[mask, 1], 
               color=class_colors[class_label], alpha=0.7, linewidth=2.5, zorder=1)
        
        # Plot points (larger)
        ax.scatter(h_2d[mask, 0], h_2d[mask, 1], 
                  color=class_colors[class_label], s=30*marker_scale, alpha=0.8,
                  edgecolors='white', linewidths=1.0, zorder=2)
        
        # Mark start point (larger)
        ax.scatter(h_2d[mask][0:1, 0], h_2d[mask][0:1, 1],
                  color=class_colors[class_label], s=150*marker_scale, marker='*',
                  edgecolors='black', linewidths=2.0, zorder=3)
    
    ax.set_title('Projection Head Space $h(z)$', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('PC 1', fontsize=12)
    ax.set_ylabel('PC 2', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add orbit variance only
    textstr = f'Mean Orbit Spread:\n{mean_var_h:.5f}\n({compression_ratio:.1f}× smaller)'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    legend_elements = [Patch(facecolor=class_colors[label], label=class_names[i]) 
                      for i, label in enumerate(orbit_data['selected_classes'])]
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                     markerfacecolor='gray', markersize=12,
                                     markeredgecolor='black', markeredgewidth=1.5,
                                     label=f'Start (0°)', linestyle='None'))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
              frameon=True, shadow=True, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle(f'Metric Singularity: Augmentation Orbit Collapse ({dataset.upper()}, {architecture})', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig3_orbit_visualization.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig3_orbit_visualization.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results from saved experiment data')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                       help='Dataset to plot results for')
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=['resnet18', 'vit_tiny'],
                       help='Architecture for collapse plot (resnet18, vit_tiny)')
    args = parser.parse_args()
    
    results_dir = f'results/{args.dataset}/{args.architecture}'
    
    if not os.path.exists(results_dir):
        print(f'Error: Results directory not found: {results_dir}')
        print('Run experiments.py first to generate results.')
        exit(1)
    
    print(f'Plotting results for {args.dataset.upper()}...')
    print()
    
    # Plot Figure 1
    if os.path.exists(f'{results_dir}/collapse_results.npy'):
        print('Creating Figure 1: Collapse Instability')
        plot_collapse_instability(results_dir, args.dataset, args.architecture)
    else:
        print('Skipping Figure 1: collapse_results.npy not found')
    
    # Plot Figure 2
    if os.path.exists(f'{results_dir}/guillotine_results.npy') and os.path.exists(f'{results_dir}/curvature_results.npy'):
        print('Creating Figure 2: Geometric Mechanisms')
        plot_geometric_mechanisms(results_dir, args.dataset, args.architecture)
    else:
        print('Skipping Figure 2: Required data files not found')
    
    # Plot Figure 3
    if os.path.exists(f'{results_dir}/orbit_visualization.npy'):
        print('Creating Figure 3: Orbit Visualization')
        plot_orbit_visualization(results_dir, args.dataset, args.architecture)
    else:
        print('Skipping Figure 3: orbit_visualization.npy not found')
    
    print()
    print('Done! Figures saved to:', results_dir)
