import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.intention_transformer import IntentionTransformer


def format_print(label, arr):
    print(f"{label}: \t" + " ".join(f"{x:2d}" for x in arr))


if __name__ == "__main__":
    from config import device
    from data_gen import data_gen

    _, test_dataset = data_gen(32)

    # Directory to save model
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(curr_dir, 'models/checkpoints')
    model = IntentionTransformer.from_pretrained(checkpoint_dir).to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Load test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Lists to store outputs and targets
    all_outputs = []
    all_targets = []
    all_internal_index = []
    all_skill_index = []
    
    with torch.no_grad():  # Disable gradient computation
        for state_seq, action_seq, skill_seq in test_loader:
            
            state_seq = torch.permute(state_seq, (0, 1, 4, 2, 3))
            state_seq = state_seq.to(device)
            action_seq = action_seq.to(device)
            
            # Get model predictions
            # TODO: delete this line after s->a translation task
            action_dummy = torch.zeros_like(action_seq).to(device)
            logits, onehot, intention_logits_for_smooth_loss, transition_logits, internal_index = model(action_dummy,
                                                                                                   state_seq,
                                                                                                   skill_seq)
            
            outputs = logits.argmax(dim=-1)
            
            # Store outputs and targets for comparison
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(action_seq.cpu().numpy())
            all_skill_index.append(skill_seq.numpy())
            all_internal_index.append(onehot.argmax(dim=-1).cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    all_internal_index = np.concatenate(all_internal_index)
    all_skill_index = np.concatenate(all_skill_index)
    
    
    # Display outputs comparison
    for i in range(len(all_outputs)):
        format_print("Target", all_targets[i])
        format_print("Output", all_outputs[i])
        format_print("Internal", all_internal_index[i])
        format_print("Skill\t", all_skill_index[i])
        print("\n")