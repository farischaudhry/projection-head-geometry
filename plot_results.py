"""
Plot results from saved experiment data.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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


def plot_collapse_instability(results_dir, dataset):
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
    
    plt.title(f'Collapse Instability ({dataset.upper()})', fontsize=12, fontweight='bold')
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


def plot_geometric_mechanisms(results_dir, dataset):
    """Plot Experiment 2: Combined Guillotine + Curvature."""
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
    plt.title(f'Probing Loss ({dataset.upper()})', fontsize=12, fontweight='bold')
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
    plt.title(f'Manifold Curvature ({dataset.upper()}, {num_seeds} seeds)',
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results from saved experiment data')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                       help='Dataset to plot results for')
    args = parser.parse_args()
    
    results_dir = f'results/{args.dataset}'
    
    if not os.path.exists(results_dir):
        print(f'Error: Results directory not found: {results_dir}')
        print('Run experiments.py first to generate results.')
        exit(1)
    
    print(f'Plotting results for {args.dataset.upper()}...')
    print()
    
    # Plot Figure 1
    print('Creating Figure 1: Collapse Instability')
    plot_collapse_instability(results_dir, args.dataset)
    
    # Plot Figure 2
    print('Creating Figure 2: Geometric Mechanisms')
    plot_geometric_mechanisms(results_dir, args.dataset)
    
    print()
    print('Done! Figures saved to:', results_dir)
