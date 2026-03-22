import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.models.intention_abstraction import IntentionAbstraction
from minigrid.models.state_embedding import StateEmbedding

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


class MaskedDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, output_dim):
        super(MaskedDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads) for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, output_dim)
        self.register_buffer("causal_mask", self.create_causal_mask(512))  # Example max length
        self._num_heads = n_heads
        
    def create_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward_segment(self, encoder_segments, repeating_mask):
        outputs = []
        # the MNA attn_mask needs to be in shape [N*num_heads, L, S] ([B*num_heads, T, T] in our case)
        T_segment = encoder_segments.size(1)
        S_segment = encoder_segments.size(2)
        device = encoder_segments.device

        # Causal mask for the segment
        causal_mask = self.causal_mask[:T_segment, :T_segment].to(device)
        causal_additive_mask = torch.where(
            causal_mask, torch.tensor(-1e9).to(device), torch.tensor(0.0).to(device)
        )  # TODO: I think this will not affect the gradient?

        # Extract repeating_mask for this segment
        repeating_additive_mask = -1e9 * (1 - repeating_mask)
        combined_mask = causal_additive_mask.unsqueeze(0) + repeating_additive_mask

        # expand for num_head times to satisfy the MHA requirement
        mask_expanded = combined_mask.unsqueeze(1).expand(-1, self._num_heads, -1, -1)  # -1 means "keep existing size"

        mask_for_mha = mask_expanded.reshape(-1, T_segment, S_segment)  # Shape: [128, 32, 32]

        for layer in self.layers:
            encoder_segments = layer(encoder_segments, encoder_segments, tgt_mask=mask_for_mha)
        outputs.append(self.output_linear(encoder_segments))

        return torch.cat(outputs, dim=1)
    
    def forward_plain(self, intentions):
        mask = self.causal_mask[:intentions.size(0), :intentions.size(0)].to(intentions.device)
        # Apply the fixed causal mask to the entire encoder output
        for layer in self.layers:
            intentions = layer(intentions, intentions, tgt_mask=mask)

        # Pass through the output linear layer
        intentions = intentions.permute((1, 0, 2))
        output = self.output_linear(intentions)
        return output
    
    
    def forward(self, intentions, repeat_mask=None):
        # Create a fixed causal mask
        self.register_buffer("causal_mask", self.create_causal_mask(512))
        if repeat_mask is not None:
            return self.forward_segment(intentions, repeat_mask)
        else:
            return self.forward_plain(intentions)
        



class IntentionTransformer(nn.Module):
    def __init__(self, input_dim, n_heads, num_layers, d_model, d_ff, num_intentions, vocab_size):
        super(IntentionTransformer, self).__init__()
        self.state_embedding = StateEmbedding(11, 6, d_model)
        self.encoder = TransformerEncoder(input_dim, n_heads, num_layers, d_model, d_ff, vocab_size)
        self.intention_head = IntentionHead(d_model, num_intentions)
        self.intention_abstraction = IntentionAbstraction(num_intentions, d_model)
        self.decoder = MaskedDecoder(d_model, n_heads, num_layers, vocab_size)
        
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
    
    def forward(self, actions, states):
        B, T, H, W, C = states.size()
        states = states.view(B * T, H, W, C)
        state_embedding = self.state_embedding(states)
        state_embedding = state_embedding.view(B, T, -1)
        encoder_output = self.encoder(actions, state_embedding)
        intentions, logits = self.intention_head(encoder_output)
        reorganized_intentions, internal_index, repeat_mask = self.intention_abstraction(intentions, encoder_output)
        reorganized_intentions = reorganized_intentions.permute((1, 0, 2))
        actions = self.decoder(reorganized_intentions, repeat_mask)
        return actions, internal_index, logits


if __name__ == "__main__":
    # Example usage
    model = IntentionTransformer(input_dim=128,
                                 n_heads=8,
                                 num_layers=6,
                                 d_model=32,
                                 d_ff=64,
                                 num_intentions=11,
                                 vocab_size=11)
    
    input_actions = torch.randint(0, 10, (32, 20))# (batch size, time steps)
    input_states = torch.rand(32, 20, 3, 11, 6) # (batch size * time steps, map_height, map_width, 3)
    output_actions = model(input_actions, input_states)