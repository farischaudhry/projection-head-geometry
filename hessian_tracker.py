import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
from experiments import get_backbone, ProjectionHead, TwoCropTransform 
from scipy.stats import spearmanr


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
    """Set Matplotlib style for double/composite plots."""
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


# Min eigenvalue estimation using Power Iteration and Hessian-Vector Products
def get_min_eigenvalue_hvp(loss, z, num_iterations=20):
    """
    Estimate the minimum eigenvalue of the Hessian w.r.t representations z 
    using Power Iteration and Hessian-Vector Products.
    """
    grad_z = torch.autograd.grad(loss, z, create_graph=True)[0]
    
    v = torch.randn_like(z)
    v = v / (torch.norm(v) + 1e-8)
    
    for _ in range(num_iterations):
        Hv = torch.autograd.grad(grad_z, z, grad_outputs=v, retain_graph=True)[0]
        v = Hv / (torch.norm(Hv) + 1e-8)
    
    lambda_max = torch.sum(v * torch.autograd.grad(grad_z, z, grad_outputs=v, retain_graph=True)[0])
    
    v_min = torch.randn_like(z)
    v_min = v_min / (torch.norm(v_min) + 1e-8)
    
    for _ in range(num_iterations):
        Hv = torch.autograd.grad(grad_z, z, grad_outputs=v_min, retain_graph=True)[0]
        shifted_Hv = Hv - lambda_max * v_min
        v_min = shifted_Hv / (torch.norm(shifted_Hv) + 1e-8)
        
    Hv_final = torch.autograd.grad(grad_z, z, grad_outputs=v_min, retain_graph=True)[0]
    lambda_min = torch.sum(v_min * Hv_final)
    
    return lambda_min.item()


def run_hessian_tracking(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Hessian Tracker on {device}")
    print(f"Config: Init={args.init.upper()}, Activation={args.activation.upper()}, Epochs={args.epochs}")
    
    batch_size = 256
    epochs = args.epochs
    activation = args.activation
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    ds = torchvision.datasets.CIFAR10(root='./data', train=True, transform=TwoCropTransform(transform), download=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    backbone = get_backbone('resnet18').to(device)
    projector = ProjectionHead(input_dim=512, activation=activation, use_bn=False).to(device)
    predictor = ProjectionHead(input_dim=2048, hidden_dim=512, output_dim=2048, activation=activation, use_bn=False).to(device)
    
    if args.init == 'collapsed':
        print("Applying Pseudo-Collapsed Weights (Head Only)...")
        with torch.no_grad():
            for m in projector.modules():
                if isinstance(m, nn.Linear): 
                    m.weight.data *= 0.1
    else:
        print("Keeping Standard (Non Pseudo-Collapsed) Weights...")

    params = list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters())
    optimizer = optim.SGD(params, lr=0.05, momentum=0.9)
    
    min_eigenvalues = []
    variances = []
    condition_numbers = [] 
    
    for epoch in range(epochs):
        epoch_eigvals = []
        epoch_vars = []
        epoch_conds = [] 
        
        for i, ((x1, x2), _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x1, x2 = x1.to(device), x2.to(device)
            
            z1, z2 = backbone(x1), backbone(x2)
            z1.requires_grad_(True) 
            
            p1, p2 = projector(z1), projector(z2)
            h1, h2 = predictor(p1), predictor(p2)
            
            loss = 0.5 * (-(F.cosine_similarity(h1, p2.detach()).mean() + F.cosine_similarity(h2, p1.detach()).mean()))
            
            if i < 5: 
                min_eig = get_min_eigenvalue_hvp(loss, z1)
                epoch_eigvals.append(min_eig)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                z_norm = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
                epoch_vars.append(torch.std(z_norm, dim=0).mean().item())
                
                z_centered = z_norm - z_norm.mean(dim=0, keepdim=True)
                cov_matrix = (z_centered.T @ z_centered) / (z_norm.size(0) - 1)
                eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                lambda_max = eigenvalues[-1].item()
                lambda_min = torch.clamp(eigenvalues[0], min=1e-7).item() 
                epoch_conds.append(lambda_max / lambda_min)
                
        min_eigenvalues.append(np.mean(epoch_eigvals))
        variances.append(np.mean(epoch_vars))
        condition_numbers.append(np.mean(epoch_conds))
        print(f"Epoch {epoch+1} | Min Eig: {min_eigenvalues[-1]:.8f} | Var: {variances[-1]:.6f} | Cond Num: {condition_numbers[-1]:.1f}")

    suffix = f"{args.init}_{args.activation}"
    np.savez(
        f'raw_data_{suffix}.npz', 
        epochs=np.arange(epochs),
        min_eigenvalues=np.array(min_eigenvalues),
        variances=np.array(variances),
        condition_numbers=np.array(condition_numbers)
    )
    print(f"\nSaved raw data to raw_data_{suffix}.npz")

    if args.plot_after_track:
        plot_individual_run(args.init, args.activation, min_eigenvalues, variances, condition_numbers, epochs)


def plot_individual_run(init, activation, min_eig, var, cond, epochs):
    single_plot_style()
    
    suffix = f"{init}_{activation}"
    epochs_range = range(len(epochs))
    
    fig, ax1 = plt.subplots() 
    ax2 = ax1.twinx()
    ax1.plot(epochs_range, min_eig, 'r-', label='Min Eigenvalue (Hessian)')
    ax2.plot(epochs_range, var, 'b--', label='Repr. Variance')
    ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Minimum Eigenvalue', color='r')
    ax2.set_ylabel('Mean Representation Variance', color='b')
    plt.title(f'Curvature Signature: {init.upper()} Init | {activation.upper()} Head')
    
    # Legend handling
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout()
    plt.savefig(f'hessian_sig_{suffix}.png', dpi=300)
    
    plt.figure() # Inherits style figsize
    plt.plot(epochs_range, cond, 'g-')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel(r'Condition Number $\kappa$')
    plt.title(f'Geometric Preconditioning: {init.upper()} Init | {activation.upper()} Head')
    plt.tight_layout()
    plt.savefig(f'cond_num_{suffix}.png', dpi=300)
    print(f"Saved individual plots for {suffix}")


def plot_hessian_composite(data_dir):
    double_plot_style()
    
    runs = [
        {"file": "raw_data_normal_swish.npz", "title": "Normal Init + Swish"},
        {"file": "raw_data_collapsed_swish.npz", "title": "Collapsed Init + Swish"},
        {"file": "raw_data_collapsed_relu.npz", "title": "Collapsed Init + ReLU"}
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Geometric Preconditioning and Collapse Recovery', fontsize=28, fontweight='bold', y=0.98)

    for col, run_info in enumerate(runs):
        filepath = os.path.join(data_dir, run_info["file"])
        if not os.path.exists(filepath):
            print(f"Warning: Could not find {filepath}. Skipping column {col+1}.")
            continue
            
        data = np.load(filepath)
        epochs = data['epochs'][1:]
        min_eig = data['min_eigenvalues'][1:]
        var = data['variances'][1:]
        cond = data['condition_numbers'][1:]

        rho_full, _ = spearmanr(var, cond)
        print(f"Full Run Spearman: {rho_full:.3f}")

        # Top row: Hessian
        ax1 = axes[0, col]
        ax2 = ax1.twinx() 
        ax1.plot(epochs, min_eig, 'r-', label='Min Eigenvalue')
        ax2.plot(epochs, var, 'b--', label='Repr. Variance')
        ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax1.set_title(run_info["title"], pad=15)
        ax1.set_xlabel('Epochs')
        ax2.set_yscale('log')        
        ax1.set_ylabel('Min Eigenvalue', color='r', fontsize=11)
        ax2.set_ylabel('Variance (Log)', color='b', fontsize=11)
        # Format ticks to be readable in log space
        ax1.tick_params(axis='y', colors='red', labelsize=10)
        ax2.tick_params(axis='y', colors='blue', labelsize=10)

        if col == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Bottom row: Condition Number
        ax3 = axes[1, col]
        ax3.plot(epochs, cond, 'g-')
        ax3.set_yscale('log')
        ax3.set_xlabel('Epochs')
        # Give every bottom plot a y-label so scales are strictly clear
        ax3.set_ylabel(r'Condition Number $\kappa$ (Log)', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(data_dir, 'hessian_tracker.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated styled master plot: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Hessian Tracker & Plotter")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='track', choices=['track', 'plot'], 
                        help="Choose 'track' to train/save data, or 'plot' to generate figures from saved data.")
    # Tracking Arguments
    parser.add_argument('--init', type=str, default='normal', choices=['normal', 'collapsed'])
    parser.add_argument('--activation', type=str, default='swish', choices=['swish', 'relu', 'gelu'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--plot_after_track', action='store_true', help="If set, generates the individual plots immediately after tracking.")
    # Plotting Arguments
    parser.add_argument('--plot_type', type=str, default='composite', choices=['composite', 'individual'],
                        help="When in 'plot' mode, choose to plot the 2x3 composite figure or re-generate individual plots.")
    parser.add_argument('--dir', type=str, default='.', help="Directory containing the .npz files for plotting.")
    args = parser.parse_args()

    if args.mode == 'track':
        run_hessian_tracking(args)
    elif args.mode == 'plot':
        print(f"Plotting mode activated. Reading data from: {os.path.abspath(args.dir)}")
        if args.plot_type == 'composite':
            plot_hessian_composite(args.dir)
        else:
            # Fallback to recreate individual plots from all .npz files in the directory
            npz_files = glob.glob(os.path.join(args.dir, "raw_data_*.npz"))
            for filepath in npz_files:
                filename = os.path.basename(filepath)
                # Parse init and activation from filename (e.g. raw_data_normal_swish.npz)
                parts = filename.replace("raw_data_", "").replace(".npz", "").split("_")
                if len(parts) == 2:
                    data = np.load(filepath)
                    plot_individual_run(parts[0], parts[1], data['min_eigenvalues'], data['variances'], data['condition_numbers'], data['epochs'])
