"""
Plot results from saved experiment data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
import os
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

# Plotting style
def single_plot_style() -> None:
    """Set Matplotlib style for single plots."""
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 28,            
        'axes.labelsize': 32,       
        'axes.titlesize': 34,       
        'xtick.labelsize': 26,      
        'ytick.labelsize': 26,      
        'legend.fontsize': 24,      
        'legend.frameon': True,
        'figure.figsize': (10, 8),  
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
    })


def double_plot_style() -> None:
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


ARCHITECTURE_TO_TITLE = {
    'resnet18': 'ResNet-18',
    'vit_tiny': 'ViT-Tiny',
}


def plot_collapse_instability(results_dir, dataset, architecture):
    """Plot Experiment 1: Main Text Figure (Linear, ReLU, GELU, Swish)."""
    single_plot_style()
    results = np.load(f'{results_dir}/collapse_results.npy', allow_pickle=True).item()
    
    plt.figure()
    
    # Map the new data keys to the original labels and colors
    runs_to_plot = {
        # --- Main Architectures ---
        'linear_bn_False_lr_base': ('LINEAR', '#1f77b4', '--'),     
        'relu_bn_False_lr_base': ('RELU (Base LR)', '#ff7f0e', '-'),
        'gelu_bn_False_lr_base': ('GELU', '#2ca02c', ':'),          
        'swish_bn_False_lr_base': ('SWISH', '#d62728', '-.'),       
        # --- ReLU Ablations (BN & Step Size) ---
        'relu_bn_True_lr_base': ('RELU (+BN)', '#9467bd', '-'),      
        'relu_bn_False_lr_large': ('RELU (Large LR)', '#8c564b', '--'), 
        'relu_bn_False_lr_small': ('RELU (Small LR)', '#e377c2', ':'),
        'relu_bn_True_lr_large': ('RELU + BN (Large LR)', '#17becf', '--'), 
        'relu_bn_True_lr_small': ('RELU + BN (Small LR)', '#bcbd22', ':'), 
    }

    max_y_val = 0  
    for run_key, (label, color, style) in runs_to_plot.items():
        if run_key in results:
            data = results[run_key]
            epochs = range(len(data['mean']))
            
            # Grab the raw (3, epochs) array
            raw_vars = np.array(data['raw'])
            
            # Calculate mean, min, and max across seeds (axis=0)
            mean = np.mean(raw_vars, axis=0)
            min_vals = np.min(raw_vars, axis=0)
            max_vals = np.max(raw_vars, axis=0)
            
            # Get normalization factor
            initial_mean = mean[0] if mean[0] > 1e-8 else 1.0 
            
            # Normalize the mean and the bounds
            mean_normalized = mean / initial_mean
            min_normalized = min_vals / initial_mean
            max_normalized = max_vals / initial_mean
            
            max_y_val = max(max_y_val, np.max(max_normalized))
            
            plt.plot(epochs, mean_normalized, 
                    label=label, 
                    linestyle=style,
                    color=color,
                    linewidth=2.5)
            
            # Fill between the absolute min and max of your 3 seeds
            plt.fill_between(epochs, 
                           min_normalized,
                           max_normalized, 
                           alpha=0.2,
                           color=color)
    
    plt.title(f'Collapse Instability ({dataset.upper()}, {architecture})', fontweight='bold', pad=15)
    plt.xlabel('Training Epochs')
    plt.ylabel('Norm. Repr. Std. Dev.')
    plt.yscale('log') 
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.legend(frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig1_collapse_instability.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig1_collapse_instability.png')


def plot_relu_gap_variance(results_dir, dataset, architecture):
    """Plot the representation variance for the ReLU/BN ablation."""
    single_plot_style()
    results = np.load(f'{results_dir}/collapse_results.npy', allow_pickle=True).item()
    
    plt.figure()
    
    runs_to_plot = {
        # --- Main Architectures ---
        'linear_bn_False_lr_base': ('LINEAR', '#1f77b4', '--'),      
        'relu_bn_False_lr_base': ('RELU (No BN)', '#ff7f0e', '-'), 
        'gelu_bn_False_lr_base': ('GELU', '#2ca02c', ':'),           
        'swish_bn_False_lr_base': ('SWISH', '#d62728', '-.'),        
        # --- ReLU Ablations (BN & Step Size) ---
        'relu_bn_True_lr_base': ('RELU (+BN)', '#9467bd', '-'),      
        'relu_bn_False_lr_large': ('RELU (Large LR)', '#8c564b', '--'), 
        'relu_bn_False_lr_small': ('RELU (Small LR)', '#e377c2', ':'),
        'relu_bn_True_lr_large': ('RELU + BN (Large LR)', '#17becf', '--'),
        'relu_bn_True_lr_small': ('RELU + BN (Small LR)', '#bcbd22', ':'),  
    }
    
    max_y_val = 0
    for run_key, (label, color, style) in runs_to_plot.items():
        if run_key in results:
            data = results[run_key]
            epochs = range(len(data['mean']))
            
            # Grab the raw (3, epochs) array
            raw_vars = np.array(data['raw'])
            
            # Calculate mean, min, and max across seeds
            mean = np.mean(raw_vars, axis=0)
            min_vals = np.min(raw_vars, axis=0)
            max_vals = np.max(raw_vars, axis=0)
            
            initial_mean = mean[0] if mean[0] > 1e-8 else 1.0
            
            # Normalize
            mean_normalized = mean / initial_mean
            min_normalized = min_vals / initial_mean
            max_normalized = max_vals / initial_mean
            
            max_y_val = max(max_y_val, np.max(max_normalized))
            
            plt.plot(epochs, mean_normalized, label=label, 
                     linestyle=style, color=color, linewidth=2.5)
            
            # Fill between absolute min and max
            plt.fill_between(epochs, min_normalized, max_normalized, alpha=0.2, color=color)
    
    plt.title(f'The ReLU Gap: Collapse Instability ({dataset.upper()}, {architecture})', fontweight='bold', pad=15)
    plt.xlabel('Training Epochs')
    plt.ylabel('Norm. Repr. Std. Dev.')
    plt.yscale('log')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.legend(frameon=True, shadow=True, loc='upper left')
    plt.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig_relu_gap.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig_relu_gap.png')


def plot_residual_gradients(results_dir, dataset, architecture):
    """Plot the gradient norms of the projection head during collapse."""
    single_plot_style()
    results = np.load(f'{results_dir}/collapse_results.npy', allow_pickle=True).item()
    
    plt.figure()
    
    runs_to_plot = {
        # --- Main Architectures ---
        'linear_bn_False_lr_base': ('LINEAR', '#1f77b4', '--'),      
        'relu_bn_False_lr_base': ('RELU (Trapped)', '#ff7f0e', '--'), 
        'gelu_bn_False_lr_base': ('GELU (Escapes)', '#2ca02c', '-'),           
        'swish_bn_False_lr_base': ('SWISH', '#d62728', '-.'),        
        # --- ReLU Ablations (BN & Step Size) ---
        'relu_bn_True_lr_base': ('RELU (+BN)', '#9467bd', '-'),      
        'relu_bn_False_lr_large': ('RELU (Large LR)', '#8c564b', '--'), 
        'relu_bn_False_lr_small': ('RELU (Small LR)', '#e377c2', ':'),
        'relu_bn_True_lr_large': ('RELU + BN (Large LR)', '#17becf', '--'),
        'relu_bn_True_lr_small': ('RELU + BN (Small LR)', '#bcbd22', ':'),  
    }
    
    for run_key, (label, color, style) in runs_to_plot.items():
        if run_key in results:
            data = results[run_key]
            epochs = range(len(data['grad_norms_mean']))
            
            # Grab the raw (3, epochs) gradient array
            raw_grads = np.array(data['grad_norms_raw'])
            
            # Calculate mean, min, and max across seeds
            grads_mean = np.mean(raw_grads, axis=0)
            grads_min = np.min(raw_grads, axis=0)
            grads_max = np.max(raw_grads, axis=0)
            
            plt.plot(epochs, grads_mean, label=label, 
                     linestyle=style, color=color, linewidth=2.5)
            
            # Fill between absolute min and max
            plt.fill_between(epochs, grads_min, grads_max, alpha=0.2, color=color)
    
    plt.title(f'Residual Gradient Norms ({dataset.upper()}, {architecture})', fontweight='bold', pad=15)
    plt.xlabel('Training Epochs')
    plt.ylabel(r'$||\nabla_h \mathcal{L}||_2$')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.legend(frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle=':')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig_gradient_norms.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig_gradient_norms.png')


def plot_condition_numbers(results_dir, dataset, architecture):
    """Plot the condition number of the representation covariance matrix."""
    single_plot_style()
    results = np.load(f'{results_dir}/collapse_results.npy', allow_pickle=True).item()
    
    plt.figure()
    
    runs_to_plot = {
        # --- Main Architectures ---
        'linear_bn_False_lr_base': ('LINEAR', '#1f77b4', '--'),      
        'relu_bn_False_lr_base': ('RELU (Trapped)', '#ff7f0e', '-'), 
        'gelu_bn_False_lr_base': ('GELU (No BN)', '#2ca02c', '--'),           
        'swish_bn_False_lr_base': ('SWISH', '#d62728', '-.'),        
        # --- ReLU Ablations (BN & Step Size) ---
        'relu_bn_True_lr_base': ('RELU (+BN)', '#9467bd', ':'),      
        'relu_bn_False_lr_large': ('RELU (Large LR)', '#8c564b', '--'), 
        'relu_bn_False_lr_small': ('RELU (Small LR)', '#e377c2', ':'),
        'relu_bn_True_lr_large': ('RELU + BN (Large LR)', '#17becf', '--'),
        'relu_bn_True_lr_small': ('RELU + BN (Small LR)', '#bcbd22', ':'),  
    }
    
    for run_key, (label, color, style) in runs_to_plot.items():
        if run_key in results:
            data = results[run_key]
            epochs = range(len(data['cond_nums_mean']))
            
            # Grab the raw (3, epochs) condition number array
            raw_conds = np.array(data['cond_nums_raw'])
            
            # Calculate mean, min, and max across seeds
            cond_mean = np.mean(raw_conds, axis=0)
            cond_min = np.min(raw_conds, axis=0)
            cond_max = np.max(raw_conds, axis=0)
            
            plt.plot(epochs, cond_mean, label=label, 
                     linestyle=style, color=color, linewidth=2.5)
            
            # Fill between absolute min and max
            plt.fill_between(epochs, cond_min, cond_max, alpha=0.2, color=color)
    
    plt.title(f'Covariance Condition Number ({dataset.upper()}, {architecture})', fontweight='bold', pad=15)
    plt.xlabel('Training Epochs')
    plt.ylabel(r'Condition Number $\kappa$')
    plt.yscale('log')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.legend(frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig_condition_numbers.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig_condition_numbers.png')


def plot_geometric_mechanisms(results_dir, dataset, architecture):
    """Plot Experiment 2b: Combined Guillotine + Curvature."""
    # Load results
    double_plot_style()
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
    
    plt.ylabel('Rotation Accuracy')
    plt.title(f'Probing Loss ({dataset.upper()}, {architecture})', fontweight='bold')
    plt.xticks(x_bar, ['Backbone (z)', 'Head (h(z))'])
    plt.ylim(0, 1.0)
    plt.axhline(0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.legend(frameon=True, shadow=True)
    plt.grid(alpha=0.3, axis='y', linestyle=':')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
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
    
    plt.ylabel('Local Curvature')
    plt.title(f'Manifold Curvature ({dataset.upper()}, {architecture}, {num_seeds} seeds)', fontweight='bold')
    plt.xticks(x, ['Backbone (z)', 'Head (h(z))'])
    plt.grid(alpha=0.3, axis='y', linestyle=':')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        std_val = [std_curv_z, std_curv_h][i]
        y_pos = height + std_val * 1.15
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.4f} ± {std_val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    textstr = f'Curvature Ratio:\n{mean_ratio:.2f}x ± {std_ratio:.2f}x'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='left', bbox=props,
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig2_geometric_mechanisms.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {results_dir}/fig2_geometric_mechanisms.png')


def _calculate_geometric_metrics(X, ids, labels):
    """High-dimensional geometric metrics calculation helper."""
    # 1. Intra-orbit distance (D_intra)
    intra_distances = []
    for oid in np.unique(ids):
        orbit_points = X[ids == oid]
        if len(orbit_points) > 1:
            dists = pdist(orbit_points, metric='euclidean')
            intra_distances.append(np.mean(dists))
    
    d_intra_mean = np.mean(intra_distances)
    d_intra_std = np.std(intra_distances)

    # 2. Inter-class distance (D_inter)
    class_centroids = []
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        class_points = X[labels == cls]
        class_centroids.append(np.mean(class_points, axis=0))
    
    class_centroids = np.array(class_centroids)
    inter_dists = pdist(class_centroids, metric='euclidean')
    d_inter_mean = np.mean(inter_dists)
    d_inter_std = np.std(inter_dists)  # Std dev of pairwise centroid distances
    
    # 3. Orbit Spread (Variance)
    variances = []
    for oid in np.unique(ids):
        orbit_points = X[ids == oid]
        centroid = np.mean(orbit_points, axis=0)
        var = np.mean(np.sum((orbit_points - centroid)**2, axis=1))
        variances.append(var)
    
    spread_mean = np.mean(variances)
    spread_std = np.std(variances)
    
    # Calculate ratio and propagate uncertainty
    ratio = d_inter_mean / (d_intra_mean + 1e-10)
    # Error propagation: if R = A/B, then σ_R = R * sqrt((sigma_A/A)^2 + (sigma_B/B)^2)
    ratio_std = ratio * np.sqrt((d_inter_std / (d_inter_mean + 1e-10))**2 + 
                                    (d_intra_std / (d_intra_mean + 1e-10))**2)
    
    return {
        'd_intra': (d_intra_mean, d_intra_std),
        'd_inter': (d_inter_mean, d_inter_std),
        'spread': (spread_mean, spread_std),
        'ratio': (ratio, ratio_std)
    }


def plot_orbit_visualization(results_dir, dataset, architecture):
    """Plot Experiment 2c: Orbit Visualization (via PCA) and calculate high-dimensional metrics."""
    # Load orbit data
    double_plot_style()
    orbit_data = np.load(f'{results_dir}/orbit_visualization.npy', allow_pickle=True).item()
    
    orbits_z = orbit_data['orbits_z']
    orbits_h = orbit_data['orbits_h']
    orbit_labels = orbit_data['orbit_labels']
    orbit_ids = orbit_data['orbit_ids']
    class_names = orbit_data['class_names']
    
    metrics_z = _calculate_geometric_metrics(orbits_z, orbit_ids, orbit_labels)
    metrics_h = _calculate_geometric_metrics(orbits_h, orbit_ids, orbit_labels)

    # Extract values for plotting
    mean_z, std_z = metrics_z['spread']
    mean_h, std_h = metrics_h['spread']
    compression_ratio = mean_z / (mean_h + 1e-10)

    # Print master table
    print('\n' + '='*60)
    print(f'High-dimensional Metrics ({dataset.upper()}, {architecture})')
    print('='*60)
    print(f"{'Metric':<25} | {'Backbone (z)':<18} | {'Head (h(z))':<18}")
    print('-'*60)
    print(f"{'Orbit Spread':<25} | {mean_z:.8f} ± {std_z:.8f} | {mean_h:.8f} ± {std_h:.8f}")
    print(f"{'Intra-orbit Dist (D_intra)':<25} | {metrics_z['d_intra'][0]:.4f} ± {metrics_z['d_intra'][1]:.4f} | {metrics_h['d_intra'][0]:.4f} ± {metrics_h['d_intra'][1]:.4f}")
    print(f"{'Inter-class Dist (D_inter)':<25} | {metrics_z['d_inter'][0]:.4f} ± {metrics_z['d_inter'][1]:.4f} | {metrics_h['d_inter'][0]:.4f} ± {metrics_h['d_inter'][1]:.4f}")
    print(f"{'Class/Orbit Ratio':<25} | {metrics_z['ratio'][0]:.2f}x ± {metrics_z['ratio'][1]:.2f}x | {metrics_h['ratio'][0]:.2f}x ± {metrics_h['ratio'][1]:.2f}x")
    print('='*60 + '\n')
    
    print(f'Orbit Compression: {compression_ratio:.2f}x (Head collapses orbits {compression_ratio:.2f}x more than Backbone)')
    print(f'Separation Improvement: {metrics_z["ratio"][0]:.2f}x → {metrics_h["ratio"][0]:.2f}x ({metrics_h["ratio"][0] / metrics_z["ratio"][0]:.2f}x better)')
    print('')
    
    # Apply PCA to reduce to 2D for visualization
    print('=== PCA PROJECTION (L2-Normalized Representations) ===')
    orbits_z_norm = orbits_z / (np.linalg.norm(orbits_z, axis=1, keepdims=True) + 1e-8)
    orbits_h_norm = orbits_h / (np.linalg.norm(orbits_h, axis=1, keepdims=True) + 1e-8)
    
    pca_z = PCA(n_components=2, random_state=42)
    z_2d = pca_z.fit_transform(orbits_z_norm)
    
    pca_h = PCA(n_components=2, random_state=42)
    h_2d = pca_h.fit_transform(orbits_h_norm)
    
    print(f'Backbone PCA explained variance: {pca_z.explained_variance_ratio_.sum():.3f}')
    print(f'Head PCA explained variance: {pca_h.explained_variance_ratio_.sum():.3f}\n')
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
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
    
    ax.set_title('Backbone Representation Space $z$', fontweight='bold', pad=15)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    textstr = f'Mean Orbit Spread:\n{mean_z:.5f} ± {std_z:.5f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
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
    
    ax.set_title('Projection Head Space $h(z)$', fontweight='bold', pad=15)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add orbit variance only
    textstr = f'Mean Orbit Spread:\n{mean_h:.5f} ± {std_h:.5f}\n({compression_ratio:.1f}× smaller)'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    legend_elements = [Patch(facecolor=class_colors[label], label=class_names[i]) 
                      for i, label in enumerate(orbit_data['selected_classes'])]
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                     markerfacecolor='gray', markersize=12,
                                     markeredgecolor='black', markeredgewidth=1.5,
                                     label=f'Start (0°)', linestyle='None'))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
              frameon=True, shadow=True, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle(f'Visualization of Augmentation Orbit Collapse ({dataset.upper()}, {architecture})', 
                fontweight='bold', y=1.02)
    
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
    architecture_title = ARCHITECTURE_TO_TITLE.get(args.architecture, args.architecture)
    
    if not os.path.exists(results_dir):
        print(f'Error: Results directory not found: {results_dir}')
        print('Run experiments.py first to generate results.')
        exit(1)
    
    print(f'Plotting results for {args.dataset.upper()}...')
    print()
    
    # Plot Figure 1
    if os.path.exists(f'{results_dir}/collapse_results.npy'):
        print('Creating Figure 1: Collapse Instability')
        plot_collapse_instability(results_dir, args.dataset, architecture_title)
        plot_relu_gap_variance(results_dir, args.dataset, architecture_title)
        plot_residual_gradients(results_dir, args.dataset, architecture_title)
        plot_condition_numbers(results_dir, args.dataset, architecture_title)
    else:
        print('Skipping Figure 1: collapse_results.npy not found')
    
    # Plot Figure 2
    if os.path.exists(f'{results_dir}/guillotine_results.npy') and os.path.exists(f'{results_dir}/curvature_results.npy'):
        print('Creating Figure 2: Geometric Mechanisms')
        plot_geometric_mechanisms(results_dir, args.dataset, architecture_title)
    else:
        print('Skipping Figure 2: Required data files not found')
    
    # Plot Figure 3
    if os.path.exists(f'{results_dir}/orbit_visualization.npy'):
        print('Creating Figure 3: Orbit Visualization')
        plot_orbit_visualization(results_dir, args.dataset, architecture_title)
    else:
        print('Skipping Figure 3: orbit_visualization.npy not found')
    
    print()
    print('Done! Figures saved to:', results_dir)
