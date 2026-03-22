import torch
from torch import nn
import torch.nn.functional as F


class IntentionAbstraction(nn.Module):
    def __init__(self, intention_size, model_dim):
        super(IntentionAbstraction, self).__init__()
    
    def segment_sequence(self, intentions, keep_feature_grad=False):
        # Find indices of intention with highest prob
        index_sequence = intentions
        if not keep_feature_grad:
            # Hard end points: find the changing points where the intention is not the same with the previous position
            # May not preserve differentiability
            boundary_labels = (index_sequence[:, 1:] != index_sequence[:, :-1]).int()
        else:
            # Soft end points
            boundary_labels = self.soft_end_dist(intentions)
        
        return boundary_labels

    def repeating(self, source, boundary_labels):
        # append one to the last position of each sequence
        ones = torch.ones((boundary_labels.shape[0], 1)).to(source.device)
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
        source = source.permute((0, 2, 1))
        feature_segments = torch.bmm(source, mask)
        feature_segments = feature_segments.permute((0, 2, 1))
        
        return feature_segments, mask
    
    def forward(self, skill_index, encoder_output, skill_embedding):
        
        # Apply the segmentation trajectory-wise
        boundary_labels = self.segment_sequence(skill_index, keep_feature_grad=False)
        
        # Get the skill embedding of the sequences
        intention_embeddings = skill_embedding
        
        # Get the repeating features within segments
        intention_features, mask = self.repeating(encoder_output, boundary_labels)
        
        # Merge the features and embedding of skills as the intention representation for decoding
        intention_merge = intention_embeddings + intention_features
        
        return intention_merge, skill_index, mask