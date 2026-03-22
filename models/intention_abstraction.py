import torch
from torch import nn
import torch.nn.functional as F


class IntentionAbstraction(nn.Module):
    def __init__(self, intention_size, model_dim):
        super(IntentionAbstraction, self).__init__()
        self.codebook = nn.Linear(intention_size, model_dim)
        self.max_num_skills = 30

    def soft_end_dist(self, intentions):
        # Compute the difference correctly
        diff = intentions[:, 1:] - intentions[:, :-1]
    
        # Create a smooth measure of change
        change_labels = diff.abs().sum(dim=-1)  # Sum of absolute differences for each time step
    
        # Use a sigmoid function to create a differentiable output
        smooth_threshold = 0.5
        change_labels = 1 / (1 + torch.exp(-10 * (change_labels - smooth_threshold)))
        return change_labels
    
    def segment(self, onehot, index, keep_feature_grad=True):
        
        if not keep_feature_grad:
            # Hard end points: find the changing points where the intention is not the same with the previous position
            # May not preserve differentiability
            boundary_labels = (index[:, 1:] != index[:, :-1]).int()
        else:
            # Soft end points
            boundary_labels = self.soft_end_dist(onehot)
        
        return boundary_labels
    
    def sampling(self, intention_embeddings, boundary_labels, index, max_length):
        batch_size, num_timesteps, num_embeddings = intention_embeddings.shape
        
        ones = torch.ones((boundary_labels.shape[0], 1)).to(intention_embeddings.device)
        boundary_labels = torch.cat([ones, boundary_labels], dim=-1)
        
        # Get lengths within a batch
        lengths = boundary_labels.to(int).sum(dim=1)
        
        # Create a mask and use it to gather elements
        mask = boundary_labels.bool()
        
        # Flatten the batch and time dimensions
        flat_features = intention_embeddings.view(-1, num_embeddings)
        flat_mask = mask.view(-1)
        
        flat_index = index.view(-1)
        
        # Get the indices of selected elements
        selected_indices = flat_mask.nonzero().squeeze(-1)
        
        # Split into batches
        split_sizes = lengths.tolist()
        selected_feature_split = torch.split_with_sizes(flat_features[selected_indices], split_sizes)
        selected_index_split = torch.split_with_sizes(flat_index[selected_indices], split_sizes)
        
        
        return selected_feature_split, selected_index_split # padded_features, padded_index

    def repeating(self, intention_embeddings, boundary_labels):
        # append one to the last position of each sequence
        ones = torch.ones((boundary_labels.shape[0], 1)).to(intention_embeddings.device)
        end_dist = torch.cat([boundary_labels, ones], dim=-1)
        
        # put the end_dist to the diagonal elements to get the end point matrix
        end_dist_M = torch.diag_embed(end_dist)
        
        # Cumulate the elements in the direction of rows reversely
        cum_sum = torch.cumsum(end_dist_M.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
        # cum_sum = torch.cumsum(end_dist_M, dim=1) # More efficient implementation of reverse cumulative sum
        # cum_sum = cum_sum[:,-1, None] - cum_sum
        
        # Cumulate the elements in the direction of columns
        cum_sum = torch.cumsum(cum_sum, dim=1)
        
        # Clamp the values in cum_sum, and use Straight-through Gradient trick to preserve differentiability
        diff = cum_sum.clamp(0, 1) - cum_sum
        cum_sum_clamp = cum_sum + diff.detach()
        
        # get masks from the shift difference of cumulative sum
        mask = cum_sum_clamp[:, 1:] - cum_sum_clamp[:, :-1]
        
        mask = torch.cat([torch.zeros_like(end_dist).unsqueeze(1), mask], dim=1)
        
        # Apply the mask to the source sequence to get the repeating segments
        intention_embeddings = intention_embeddings.permute((0, 2, 1))
        feature_segments = torch.bmm(intention_embeddings, mask)
        feature_segments = feature_segments.permute((0, 2, 1))
        
        return feature_segments, mask
    
    def forward(self, onehot, index, encoder_output):
        
        # Apply the segmentation trajectory-wise
        boundary_labels = self.segment(onehot, index, keep_feature_grad=False)
        
        # Get the skill embedding of the sequences
        intention_embeddings = self.codebook(onehot)
        
        # Get the skill sequence for n-gram transition leaning
        intention_embedding_corpus, intention_index_corpus = self.sampling(intention_embeddings,
                                                                                  boundary_labels,
                                                                                  index,
                                                                                  max_length=self.max_num_skills)
        
        # Get the repeating features within segments
        intention_features, mask = self.repeating(encoder_output, boundary_labels)
        
        # Merge the features and embedding of skills as the intention representation for decoding
        intention_merge = intention_embeddings + intention_features
        
        return intention_merge, mask, intention_embedding_corpus, intention_index_corpus