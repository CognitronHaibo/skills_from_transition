import torch


def repeating(source, end_dist):
    # append one to the last position of each sequence
    end_dist = torch.cat([end_dist, torch.ones((end_dist.shape[0], 1))], dim=-1)
    
    # put the end_dist to the diagonal elements to get the end point matrix
    end_dist_M = torch.diag_embed(end_dist)
    
    # Cumulate the elements in the direction of rows reversely
    cum_sum = torch.cumsum(end_dist_M.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
    # cum_sum = torch.cumsum(end_dist_M, dim=1)
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
    result = torch.bmm(source, mask)
    # result = (mask.unsqueeze(-1) @ source)
    
    return result


if __name__ == "__main__":
    end_dist = torch.tensor([[0, 0, 1, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 1, 0, 0]], dtype=torch.float32)
    source = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],
                           [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]], dtype=torch.float32)
    print(repeating(source, end_dist))