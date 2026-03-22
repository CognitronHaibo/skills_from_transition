import numpy as np
import torch
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle
import os



class SequenceDataset(Dataset):
    def __init__(self, state_sequences, action_sequences, skill_sequences):
        """
        Args:
            state_sequences (list of np.ndarray): List of source sequences.
            action_sequences (list of np.ndarray): List of target sequences (dummy data).
        """
        self.state_sequences = state_sequences
        self.action_sequences = action_sequences
        self.skill_sequences = skill_sequences

    def __len__(self):
        return len(self.state_sequences)

    def __getitem__(self, idx):
        # Convert sequences to PyTorch tensors
        state_seq = torch.tensor(self.state_sequences[idx]).float()
        action_seq = torch.tensor(self.action_sequences[idx], dtype=int)
        skill_seq = torch.tensor(self.skill_sequences[idx], dtype=int)
        return state_seq, action_seq, skill_seq

def data_gen(max_length):

    curr_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(curr_dir, r"datasets/ninerooms/action_state_trajectories.pkl")
    with open(data_dir, 'rb') as file:
        traj_dict = pickle.load(file)
    
    action_sequences = []
    state_sequences = []
    skill_sequences = []
    for ind, traj, in traj_dict.items():
        
        # shift the action value to 1-7, with 0 for padding
        action_traj = [a + 1 for a, state, skill in traj]
        state_traj = [state for a, state, skill in traj]
        skill_traj = [skill for a, state, skill in traj] # Add 1 if the skill is the original index

        state_shape = state_traj[0].shape
        
        # Pad the sequence if it's shorter than max_length
        if len(action_traj) < max_length:
            padding_action = np.full(max_length - len(action_traj), 0)  # Use 0 as padding value
            action_traj = np.concatenate((action_traj, padding_action))

            
            padding_state = np.full((max_length - len(state_traj), *state_shape), 0)  # Use 0 as padding value
            state_traj = np.concatenate((np.array(state_traj), padding_state), axis=0, dtype=float)

            skill_traj = np.concatenate((skill_traj, padding_action))

        action_sequences.append(action_traj)
        state_sequences.append(state_traj)
        skill_sequences.append(skill_traj)
        
        
        
    # Split the training and test data
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(state_sequences, action_sequences, skill_sequences,
                                                                         test_size=0.2, random_state=42)
    
    
    # Create the dataset
    train_dataset = SequenceDataset(X_train, y_train, z_train)
    test_dataset = SequenceDataset(X_test, y_test, z_test)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    sequences, _ = data_gen(32)
    for state_traj, action_traj in sequences:
        print(state_traj.shape)
        break
    
        
