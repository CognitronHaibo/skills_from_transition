import os
import json
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder
import torch.nn.functional as F
from models.intention_abstraction import IntentionAbstraction
from models.state_embedding import StateEmbedding
from models.causal_decoding_layer import CausalDecoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, n_heads, num_layers, d_model, d_ff, vocab_size):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)
        ])

    def sinusoidal_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        dimension = torch.arange(d_model, dtype=torch.float).unsqueeze(0)

        # Apply the sine and cosine functions
        angle_rates = 1 / torch.pow(10000, (2 * (dimension // 2)) / d_model)
        angle_rads = position * angle_rates

        # Apply sine to even indices and cosine to odd indices
        pos_enc = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return pos_enc

    def forward(self, actions, states):
        max_len = actions.size(1)
        pos_enc = self.sinusoidal_positional_encoding(max_len, self.d_model).to(actions.device)
        actions = self.embedding(actions) + pos_enc + states
        for layer in self.layers:
            actions = layer(actions)
        return actions


class IntentionHead(nn.Module):
    def __init__(self, d_model, num_intentions):
        super(IntentionHead, self).__init__()
        self.fc = nn.Linear(d_model, num_intentions)

    def forward(self, x):
        logits = self.fc(x)
        return F.gumbel_softmax(logits, tau=1.0, hard=True), logits


class MaskedDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, d_model, output_dim, norm=None):
        super(MaskedDecoder, self).__init__(decoder_layer, num_layers, norm)
        self.output_linear = nn.Linear(d_model, output_dim)
        self.register_buffer("causal_mask", self.create_causal_mask(512))  # Example max length

    def create_causal_mask(self, size):
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
        return mask

    def forward_segment(self, encoder_segments, repeating_mask):
        raise NotImplementedError

    def forward_plain(self, tgt, memory,
                      tgt_mask=None, memory_mask=None,
                      tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        # Apply the fixed causal mask to the entire encoder output
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        # Pass through the output linear layer
        output = output.permute((1, 0, 2))
        output = self.output_linear(output)
        return output


    def forward(self, intentions, segmentwise=False):
        # Create a fixed causal mask
        mask = self.causal_mask[:intentions.size(0), :intentions.size(0)].to(intentions.device)
        if segmentwise:
            return self.forward_segment(intentions)
        else:
            return self.forward_plain(intentions, intentions,
                      tgt_mask=mask, memory_mask=mask)


class IntentionTransformer(nn.Module):
    def __init__(self, input_dim, n_heads, num_layers, d_model, d_ff, num_intentions, vocab_size):
        super(IntentionTransformer, self).__init__()
        self.state_embedding = StateEmbedding(13, 13, d_model)
        self.encoder = TransformerEncoder(input_dim, n_heads, num_layers, d_model, d_ff, vocab_size)

        # Additional components for skill processing and decoding
        self.skill_embedding = nn.Embedding(num_intentions, d_model)
        self.action_embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff)
        self.decoder = MaskedDecoder(decoder_layer, num_layers=num_layers, d_model=d_model, output_dim=vocab_size)
        self.d_model = d_model

        # Layer normalization for skill features
        self.layer_norm = nn.LayerNorm(d_model)

        # Store parameters for saving
        self.config = {
            'input_dim': input_dim,
            'n_heads': n_heads,
            'num_layers': num_layers,
            'd_model': d_model,
            'd_ff': d_ff,
            'num_intentions': num_intentions,
            'vocab_size': vocab_size
        }

    def sinusoidal_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        dimension = torch.arange(d_model, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (dimension // 2)) / d_model)
        angle_rads = position * angle_rates
        pos_enc = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return pos_enc

    def prepending(self, tokens):
        BOS_token_index = 8  # 8 for <bos> token in addition to 7 minigrid actions
        batch_size = tokens.shape[0]
        bos_prepending = torch.full((batch_size, 1), BOS_token_index, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([bos_prepending, tokens[:, :-1]], dim=-1)

    def forward(self, actions, states, skills, mute_diagonal=False):
        """
        actions: tensor with shape (B, T)
        states: tensor with shape (B, T, C, H, W)
        skills: tensor with shape (B, T)
        """
        B, T, C, H, W = states.size()
        states = states.view(B * T, C, H, W)  # getting (B*T, C, H, W)
        state_embedding = self.state_embedding(states)  # (B*T, d_model)
        state_embedding = state_embedding.view(B, T, -1)  # (B, T, d_model)

        # Get causal mask
        causal_mask = self.decoder.causal_mask[:T, :T].to(actions.device)
        target_mask = deepcopy(causal_mask)
        n = target_mask.size(0)
        # Compare two ways of causal attention
        if mute_diagonal:
            # TODO: set the diagnola elements to -inf to implement shifting target input
            target_mask[range(n), range(n)] = float('-inf')
        else:
            # TODO: Shift the action to right by prepending zeros before action sequences
            actions = self.prepending(actions)
            encoder_output = self.encoder(actions, state_embedding)  # (B, T, d_model)
        
        


        skill_feats = self.skill_embedding(skills)  # (B, T, d_model)

        # Positional encoding
        pos_enc = self.sinusoidal_positional_encoding(T, self.d_model).to(actions.device)  # (T, d_model)
        # skill_feats += pos_enc

        skill_feats = self.layer_norm(skill_feats)  # Apply layer normalization

        # Compute attention scores
        attn_scores = torch.bmm(skill_feats, encoder_output.transpose(1, 2)) / (self.d_model ** 0.5)  # (B, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)

        encoded_skill_embedding = torch.bmm(attn_weights, encoder_output)  # (B, T, d_model)

        # Positional encoding
        pos_enc = self.sinusoidal_positional_encoding(T, self.d_model).to(actions.device)  # (T, d_model)

        action_emb = self.action_embedding(actions)  # (B, T, d_model) actions[:, 1:T] ...
        tgt = action_emb + pos_enc.unsqueeze(0)  # (B, T, d_model)
        tgt = tgt.permute(1, 0, 2)  # (T, B, d_model)

        memory = encoded_skill_embedding.permute(1, 0, 2)  # (T, B, d_model)

        
        

        # decoded_logits = self.decoder.forward_plain(memory, memory, tgt_mask=causal_mask, memory_mask=causal_mask)  # (B, T, vocab_size)
        decoded_logits = self.decoder.forward_plain(tgt, memory, tgt_mask=target_mask, memory_mask=causal_mask)  # (B, T, vocab_size)

        return decoded_logits, skills, encoder_output

    def save_model(self, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save(self.state_dict(), model_path)

        # Save model parameters to JSON
        config_path = os.path.join(checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        print(f'Model and config saved to {checkpoint_dir}')

    @classmethod
    def from_pretrained(cls, checkpoint_dir):
        # Load model parameters from JSON
        config_path = os.path.join(checkpoint_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create an instance of the model with the loaded parameters
        model = cls(**config)  # Unpack parameters
        model_path = os.path.join(checkpoint_dir, 'model.pth')
        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')
        return model