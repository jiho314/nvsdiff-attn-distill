import torch
import pickle
import os
from torchvision.utils import save_image, make_grid

path = '/mnt/data1/jiho/vggt-nvs/MVGenMaster/check_points/cat3d_distill/nan_batch_rank1.pkl'
ho = pickle.load(open(path,'rb'))
image = ho['batch']['image']  # B V C H W

# Create output directory
output_dir = '/mnt/data1/jiho/vggt-nvs/MVGenMaster/debug_images'
os.makedirs(output_dir, exist_ok=True)

print(f"Image shape: {image.shape}")
print(f"Image min: {image.min()}, max: {image.max()}")
print(f"Has NaN: {torch.isnan(image).any()}")
print(f"Has Inf: {torch.isinf(image).any()}")

# Reshape images for grid: (B*V, C, H, W)
B, V, C, H, W = image.shape
images_flat = image.reshape(B * V, C, H, W)

# Clamp values to [0, 1] range
images_clamped = torch.clamp(images_flat, 0, 1)

# Create grid with V columns (one row per batch, one column per view)
grid = make_grid(images_clamped, nrow=V, padding=2, pad_value=1.0)

# Save grid
grid_path = os.path.join(output_dir, 'image_grid.png')
save_image(grid, grid_path)

print(f"Grid saved to {grid_path}")
print(f"Grid shape: {grid.shape}")
print(f"Grid layout: {B} batches x {V} views")

pred_attn_logit = ho['pred_attn_logit']
gt_attn_logit = ho['gt_attn_logit']

def visualize_attention_map(attn_logit, query_x, query_y, title_prefix="attention", save_dir=None):
    """
    Visualize attention map for a given query coordinate
    
    Args:
        attn_logit: attention logits tensor, shape (B, Head, HW, VHW)
        query_x: x coordinate of query point
        query_y: y coordinate of query point  
        title_prefix: prefix for saved filename
        save_dir: directory to save images
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if save_dir is None:
        save_dir = output_dir
    
    print(f"Attention logit shape: {attn_logit.shape}")
    print(f"Query point: ({query_x}, {query_y})")
    
    # Handle shape (B, Head, HW, VHW)
    B, Head, HW, VHW = attn_logit.shape
    
    # Calculate H, W, V from HW and VHW
    # Assuming square images, so H = W = sqrt(HW)
    H = W = int(HW ** 0.5)
    V = VHW // HW
    
    print(f"Derived dimensions: B={B}, Head={Head}, H={H}, W={W}, V={V}")
    print(f"HW={HW}, VHW={VHW}")
    
    # Average over head dimension
    attn_avg = attn_logit.mean(dim=1)  # (B, HW, VHW)
    
    for b in range(B):
        attn_batch = attn_avg[b]  # (HW, VHW)
        
        # Convert query coordinates to linear index
        query_idx = query_y * W + query_x
        
        if query_idx >= HW:
            print(f"Query point ({query_x}, {query_y}) is out of bounds for H={H}, W={W}")
            continue
            
        # Get attention for the specific query point
        query_attn = attn_batch[query_idx]  # (VHW,)
        
        # Reshape to (V, H, W) then to (H, V*W) for visualization
        query_attn_reshaped = query_attn.reshape(V, H, W)  # (V, H, W)
        query_attn_vis = query_attn_reshaped.permute(1, 0, 2).reshape(H, V * W)  # (H, V*W)
        
        # Convert to numpy and normalize
        attn_np = query_attn_vis.detach().cpu().numpy()
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.imshow(attn_np, cmap='hot', interpolation='bilinear', aspect='auto')
        plt.colorbar()
        plt.title(f'{title_prefix} - Batch {b}\nQuery: ({query_x}, {query_y}) -> idx {query_idx}\nShape: {H} x {V*W} (H x V*W)')
        plt.xlabel(f'V*W (V={V}, each view has W={W})')
        plt.ylabel(f'H={H}')
        
        # Add vertical lines to separate views
        for v in range(1, V):
            plt.axvline(x=v*W - 0.5, color='white', linewidth=2, alpha=0.7)
        
        # Save
        save_path = os.path.join(save_dir, f'{title_prefix}_b{b}_q{query_x}_{query_y}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")
        
        # Also save a summary showing max attention per view
        plt.figure(figsize=(8, 4))
        max_attn_per_view = query_attn_reshaped.reshape(V, -1).max(dim=1)[0]
        plt.bar(range(V), max_attn_per_view.detach().cpu().numpy())
        plt.title(f'{title_prefix} - Max Attention per View - Batch {b}\nQuery: ({query_x}, {query_y})')
        plt.xlabel('View Index')
        plt.ylabel('Max Attention')
        
        summary_path = os.path.join(save_dir, f'{title_prefix}_summary_b{b}_q{query_x}_{query_y}.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved summary {summary_path}")

# Example usage: visualize attention at query point (64, 64)
query_x, query_y = 10, 10
visualize_attention_map(gt_attn_logit, query_x, query_y, "gt_attention", output_dir)
visualize_attention_map(pred_attn_logit, query_x, query_y, "pred_attention", output_dir)