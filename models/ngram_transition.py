import torch
from torch import nn


class TransitionModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, max_length, num_heads=2, num_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.max_length = max_length
        
        # Index for special tokens
        self.bos_index = torch.tensor(num_embeddings + 1, dtype=torch.long)
        self.eos_index = torch.tensor(num_embeddings + 2, dtype=torch.long)
        self.pad_index = torch.tensor(0, dtype=torch.long)
        
        # Embedding for special tokens
        self.bos_embed = nn.Parameter(torch.randn(embedding_dim))  # [D]
        self.eos_embed = nn.Parameter(torch.randn(embedding_dim))  # [D]
        
        # Positional embeddings (now accounts for BOS/EOS)
        self.pos_embed = nn.Parameter(torch.randn(1, 102, embedding_dim))  # Max 100 skills + BOS + EOS
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Prediction head (now predicts EOS as an additional output)
        self.head = nn.Linear(embedding_dim, num_embeddings + 3)  # + 3 for BOS, EOS, PAD
    
    def forward(self, selected_feature_split, selected_index_split):
        
        feature_split_eos = []
        index_split_eos = []
        for feat_seq, idx_seq in zip(selected_feature_split, selected_index_split):
            # Append eos token to features
            feat = torch.cat([
                feat_seq,
                self.eos_embed.unsqueeze(0)  # add dimension for sequence length
            ], dim=0)
            feature_split_eos.append(feat)
            
            # Append eos token index to indices
            idx = torch.cat([
                idx_seq,
                torch.tensor([self.eos_index], device=idx_seq.device)
            ], dim=0)
            index_split_eos.append(idx)
            
        # Pad each sequence to max_length
        features_pad = torch.nn.utils.rnn.pad_sequence(
            feature_split_eos,
            batch_first=True,
            padding_value=0.0
        )[:, :self.max_length]
        
        index_pad = torch.nn.utils.rnn.pad_sequence(
            index_split_eos,
            batch_first=True,
            padding_value=self.pad_index
        )[:, :self.max_length]
        
        
        # skill_embeddings: [B, K, D]
        B, K, D = features_pad.shape
        
        # Prepend BOS embeddings, only for the input sequence
        bos = self.bos_embed.repeat(B, 1, 1)  # [B, 1, D]
        x = torch.cat([bos, features_pad], dim=1)  # [B, K+1, D]
        
        
        # Truncate the input before last position
        x = x[:, :-1]# [B, K, D]
        
        # Positional embeddings
        positions = self.pos_embed[:, :K, :]  # [1, K, D]
        x = x + positions
        
        # Autoregressive mask (now includes BOS/EOS)
        seq_len = K
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)
        
        # Transformer decoder
        h = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=mask,
            memory_mask=None
        )  # [B, K, D]
        
        # Predict logits
        logits = self.head(h)  # [B, K+2, num_embeddings + 1]
        
        # Step 5 Prepare the label tensor
        label = index_pad # [B, K]
        
        return logits, label