import torch

def soft_end_dist(intentions):
    # Compute the difference correctly
    diff = intentions[:, 1:] - intentions[:, :-1]
    
    # Create a smooth measure of change
    change_labels = diff.abs().sum(dim=-1)  # Sum of absolute differences for each time step
    
    # Use a sigmoid function to create a differentiable output
    smooth_threshold = 0.1
    change_labels = 1 / (1 + torch.exp(-10 * (change_labels - smooth_threshold)))
    return change_labels