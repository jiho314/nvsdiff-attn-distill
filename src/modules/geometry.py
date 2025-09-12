import math
import torch

def cycle_consistency_checker(costmap, pixel_threshold=None):
    ''' costmap : (B, HW, VHW)
    '''
    # Get dimensions from the input tensor
    B, HW, _ = costmap.shape
    device = costmap.device

    H = W = int(math.sqrt(HW))

    # Step 2: Find the best initial matches (already vectorized)
    max_idx = torch.argmax(costmap, dim=-1)  # (B, HW)
    transpose_costmap = costmap.transpose(1, 2)  # (B, 2HW, HW)
    b_idx = torch.arange(B, device=device)[:, None] # (B, 1)
    reverse_map = transpose_costmap[b_idx, max_idx] # (B, HW, HW)
    _, final_indices = torch.max(reverse_map, dim=-1) 
    # Create a tensor representing the original indices (0, 1, 2, ..., HW-1) for each batch item
    original_indices = torch.arange(HW, device=device).expand(B, -1) # (B, HW)

    # Convert 1D indices to 2D coordinates for both original and matched points
    x1 = original_indices // W
    y1 = original_indices % H
    x2 = final_indices // W
    y2 = final_indices % H

    if pixel_threshold is None:
        pixel_threshold = round(H / 10.0)

    # Calculate Euclidean distance and check if it's within the threshold
    distance = torch.sqrt(((x1 - x2)**2 + (y1 - y2)**2).float())
    is_close = (distance < pixel_threshold).float() # .float() converts boolean (True/False) to (1.0/0.0)

    # Step 6: Reshape final output tensor
    final_distance_tensor = is_close.view(B, HW, 1)

    return final_distance_tensor