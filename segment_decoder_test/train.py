import os
from torch.functional import F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits, targets, mask):
        # Reshape logits and targets
        logits_reshaped = logits.view(-1, logits.size(-1))  # Shape: (batch_size * time_steps, vocab_size)
        targets_reshaped = targets.view(-1)  # Shape: (batch_size * time_steps,)
        
        # Compute the loss
        loss = self.criterion(logits_reshaped, targets_reshaped)
        
        # Apply the mask: only include valid entries in the loss
        valid_loss = loss * mask.view(-1)  # Shape: (batch_size * time_steps,)
        
        # Compute the mean loss over valid entries
        return valid_loss.sum() / mask.sum() if mask.sum() > 0 else loss.new_zeros(())


class TemporalSmoothCELoss(nn.Module):
    def __init__(self):
        super(TemporalSmoothCELoss, self).__init__()
    
    def forward(self, logits):
        B, T, C = logits.shape
        if T <= 1:
            return torch.tensor(0.0).to(logits.device)
        
        # Extract consecutive logits
        logits_shifted = logits[:, 1:, :]  # Shape: (B, T-1, C)
        logits_prev = logits[:, :-1, :]  # Shape: (B, T-1, C)
        target_idx = logits_prev.argmax(dim=-1)  # Shape: (B, T-1)
        return F.cross_entropy(logits_shifted.reshape(-1, C), target_idx.reshape(-1))


def train(model, data_loader, optimizer, criterion, num_epochs, device, checkpoint_dir):
    model = model.to(device)
    
    # Set the model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for state_seq, action_seq, skill_seq in data_loader:
            state_seq = torch.permute(state_seq, (0, 1, 4, 2, 3))
            state_seq = state_seq.to(device)
            action_seq = action_seq.to(device)
            skill_seq = skill_seq.to(device)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(skill_seq, state_seq, action_seq)
            
            # Compute the loss
            mask = action_seq != 0
            recon_loss = criterion["recon_loss"](logits, action_seq, mask)
            loss = recon_loss
            
            # Cumulate total loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()  # Update weights
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')
    
    # Save model
    model.save_model(checkpoint_dir)


# Example seq2seq model, optimizer, and loss function (placeholder)
from segment_decoder_test.segment_decoder import IntentionTransformer

if __name__ == "__main__":
    from config import device
    # Import training and test dataset
    from data_gen import data_gen
    
    train_dataset, _ = data_gen(32)
    
    # Directory to save model
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(curr_dir, 'checkpoints')
    
    resume = True
    
    # Initialize model, optimizer, and loss function
    if resume:
        model = IntentionTransformer.from_pretrained(checkpoint_dir)
    else:
        model = IntentionTransformer(input_dim=128,
                                     n_heads=8,
                                     num_layers=6,
                                     d_model=512,
                                     d_ff=64,
                                     num_intentions=10,
                                     vocab_size=10,
                                     vocab_size_skill=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Masked CE loss function to ignore the loss from padding elements
    criterion = {"recon_loss": MaskedCrossEntropyLoss(),
                 "smooth_loss": TemporalSmoothCELoss()}
    
    # Create a DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    num_epochs = 160
    train(model, train_loader, optimizer, criterion, num_epochs, device, checkpoint_dir)



