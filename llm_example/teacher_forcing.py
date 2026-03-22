import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Causal mask to prevent attending to future positions
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(1, 1, 2048, 2048), diagonal=1).bool()
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v
        
        # Reshape and project back to original dimension
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query projections (for decoder inputs)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # Key, Value projections (for encoder outputs)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, encoder_output):
        batch_size, seq_len, _ = x.shape
        
        # Project queries from decoder inputs
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Project keys and values from encoder outputs
        k = self.k_proj(encoder_output)  # [batch_size, enc_seq_len, embed_dim]
        v = self.v_proj(encoder_output)  # [batch_size, enc_seq_len, embed_dim]
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v
        
        # Reshape and project back to original dimension
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.self_attn = CausalSelfAttention(embed_dim, num_heads)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, encoder_output):
        # Self attention with residual connection
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)
        
        # Cross attention with encoder outputs
        cross_output = self.cross_attn(x, encoder_output)
        x = self.norm2(x + cross_output)
        
        # Feed forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(2048, embed_dim)  # Max sequence length
        
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, target, encoder_output):
        # Create position ids
        seq_len = target.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=target.device)
        position_ids = position_ids.unsqueeze(0).expand_as(target)
        
        # Embed tokens and positions
        token_embeddings = self.token_embedding(target)
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output)
        
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits


# Example usage
if __name__ == "__main__":
    # Configuration
    embed_dim = 512
    num_heads = 8
    hidden_dim = 2048
    num_layers = 6
    vocab_size = 10000
    batch_size = 4
    seq_len = 32
    enc_seq_len = 64
    
    # Create model
    decoder = TransformerDecoder(embed_dim, num_heads, hidden_dim, num_layers, vocab_size)
    
    # Dummy inputs
    encoder_output = torch.randn(batch_size, enc_seq_len, embed_dim)
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = decoder(target, encoder_output)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, seq_len, vocab_size]