import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, d_model, num_intentions, activation=None):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(1, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, num_intentions)
        
        self.activation = activation
        
        # Store parameters for saving
        self.config = {
            'd_model': d_model,
            'num_intentions': num_intentions
        }
    
    def forward(self, states):
        logits = self.fc3(self.fc2(self.fc(states.unsqueeze(-1).type(torch.float32))))
        if self.activation is None:
            return logits
        elif self.activation == 'softmax':
            return F.softmax(logits, dim=-1)
        elif self.activation == 'gumble_softmax':
            return F.gumbel_softmax(logits, tau=1.0, hard=True)
        else:
            raise NotImplementedError


def temporal_smoothness_base(logits, loss_fn):
    B, T, C = logits.shape
    if T <= 1:
        return torch.tensor(0.0).to(logits.device)
    
    # Extract consecutive logits
    logits_shifted = logits[:, 1:, :]  # Shape: (B, T-1, C)
    logits_prev = logits[:, :-1, :]  # Shape: (B, T-1, C)
    return loss_fn(logits_shifted, logits_prev)


def temporal_smoothness_cross_entropy(logits):
    B, T, C = logits.shape
    if T <= 1:
        return torch.tensor(0.0).to(logits.device)
    
    # Extract consecutive logits
    logits_shifted = logits[:, 1:, :]  # Shape: (B, T-1, C)
    logits_prev = logits[:, :-1, :]  # Shape: (B, T-1, C)
    target_idx = logits_prev.argmax(dim=-1)  # Shape: (B, T-1)
    return F.cross_entropy(logits_shifted.reshape(-1, C), target_idx.reshape(-1))


def temporal_smoothness_l1_detach_prev(logits):
    B, T, C = logits.shape
    if T <= 1:
        return torch.tensor(0.0).to(logits.device)
    
    # Extract consecutive logits
    logits_shifted = logits[:, 1:, :]  # Shape: (B, T-1, C)
    logits_prev = logits[:, :-1, :].detach()  # Shape: (B, T-1, C)
    return F.l1_loss(logits_shifted, logits_prev)


def temporal_smoothness_l1(logits):
    return temporal_smoothness_base(logits, F.l1_loss)


def temporal_smoothness_mse(logits):
    return temporal_smoothness_base(logits, F.mse_loss)


def const_target_l1_loss(logits):
    target_tensor = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ])
    target_tensor = target_tensor.repeat(32, 20, 1).to(logits.device)
    
    return F.l1_loss(logits, target=target_tensor)


def train(model, data_loader, optimizer, num_epochs, device, loss_func, check_dir):
    model = model.to(device)
    model.train()  # Set the model to training mode
    
    output_argmax = []
    for epoch in range(num_epochs):
        total_loss = 0
        
        for idx, (source_seq, target_seq) in enumerate(data_loader):
            source_seq = source_seq.to(device)
            
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            intentions = model(source_seq)  # Replace with actual model call
            
            # Compute the loss
            loss = 10 * loss_func(intentions)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()  # Update weights
            
            if (epoch == 0 and idx == 0) or (epoch == num_epochs - 1 and idx == 0):
                output_argmax.append(intentions[0].argmax(dim=-1))
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')
    
    print("Before Training:", output_argmax[0].detach().cpu().numpy())
    print("After Training:", output_argmax[1].detach().cpu().numpy())


# TODO 是否有梯度
def check_one_model_loss_setting(model_activation, loss_fn_name, num_epochs=1):
    model_activation_str = "logits" if model_activation is None else model_activation
    print(f"!!! TESTING model: {model_activation_str}, loss: {loss_fn_name}")
    model = SimpleNet(d_model=32, num_intentions=10, activation=model_activation)
    loss_func = globals()[loss_fn_name]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create a DataLoader
    batch_size = 32  # Set your batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    train(model, train_loader, optimizer, num_epochs, device, loss_func, "models/checkpoints_softmax")


if __name__ == "__main__":
    from config import device
    from data_gen import train_dataset
    from torch.utils.data import DataLoader
    
    activations = [None,
                   "softmax",
                   "gumble_softmax"]
    
    loss_fn_names = ["const_target_l1_loss",
                     "temporal_smoothness_cross_entropy",
                     "temporal_smoothness_l1",
                     "temporal_smoothness_mse",
                     "temporal_smoothness_l1_detach_prev"]
    
    # activations = ["gumble_softmax"]
    # loss_fn_names = ["const_target_l1_loss", "temporal_smoothness_cross_entropy", "temporal_smoothness_l1", "temporal_smoothness_mse", "temporal_smoothness_l1_detach_prev"]
    
    for activation in activations:
        for loss_fn_name in loss_fn_names:
            check_one_model_loss_setting(activation, loss_fn_name, 32)

