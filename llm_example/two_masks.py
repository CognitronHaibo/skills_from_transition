import torch
import torch.nn.functional as F

sz = 3
# 方法一
mask_inf = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
# 方法二
mask_bool = torch.triu(torch.ones(sz, sz), diagonal=1).bool()

# 模拟注意力权重
attn_weights = torch.randn(sz, sz)
# 方法一处理
output1 = F.softmax(attn_weights + mask_inf, dim=-1)
# 方法二处理
output2 = F.softmax(attn_weights.masked_fill(mask_bool, float('-inf')), dim=-1)

print(torch.allclose(output1, output2))  # 输出应为True