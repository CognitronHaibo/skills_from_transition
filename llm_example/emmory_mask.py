import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class CausalDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 自注意力部分 (保持原有的causal attention)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 对encoder_output也使用causal attention
        # 如果memory_mask未提供，我们创建一个causal mask
        if memory_mask is None:
            sz = memory.size(0)
            memory_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(memory.device)
        
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN部分保持不变
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class CustomTransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


# 使用示例
d_model = 512
nhead = 8
num_decoder_layers = 6

# 创建自定义decoder
decoder_layer = CausalDecoderLayer(d_model, nhead)
decoder = CustomTransformerDecoder(decoder_layer, num_decoder_layers)

# 假设我们有一些输入数据
batch_size = 4
seq_len = 10
tgt = torch.rand(seq_len, batch_size, d_model)  # decoder输入
memory = torch.rand(seq_len, batch_size, d_model)  # encoder输出

# 前向传播
output = decoder(tgt, memory)

print(output.shape)  # 应该输出: torch.Size([10, 4, 512])