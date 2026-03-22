import os
import json
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder
import torch.nn.functional as F
from segment_decoder_test.intention_abstraction import IntentionAbstraction
from models.state_embedding import StateEmbedding
from models.causal_decoding_layer import CausalDecoderLayer


class DirectEncoder(nn.Module):
    def __init__(self, input_dim, n_heads, num_layers, d_model, d_ff, vocab_size_skill):
        super(DirectEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size_skill, d_model)
    
    @classmethod
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
    
    def forward(self, skills, states):
        max_len = skills.size(1)
        pos_enc = self.sinusoidal_positional_encoding(max_len, self.d_model).to(skills.device)
        skill_embedding = self.embedding(skills)
        encoder_output = skill_embedding + pos_enc + states
        return encoder_output,  skill_embedding


class MaskedDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, d_model, vocab_size, norm=None):
        super(MaskedDecoder, self).__init__(decoder_layer, num_layers, norm)
        self.d_model = d_model
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.register_buffer("causal_mask", self.create_causal_mask(512))  # Example max length
        self.action_embedding = nn.Embedding(vocab_size, d_model)
    
    def create_causal_mask(self, size):
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
        return mask
    
    def tgt_embedding(self, target):
        # Embed target tokens and positions
        token_embeddings = self.action_embedding(target)
        seq_len = target.size(1)
        
        # Apply positional encoding from Encoder class
        position_embeddings = DirectEncoder.sinusoidal_positional_encoding(seq_len, self.d_model).to(target.device)
       
        return token_embeddings + position_embeddings
    
    def forward_segment(self, encoder_segments, repeating_mask):
        raise NotImplementedError
    
    def forward_plain(self, tgt, memory,
                      with_tgt=False,
                      tgt_mask=None, memory_mask=None,
                      tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if with_tgt:
            tgt = self.tgt_embedding(tgt)
            output = tgt
            # Apply the fixed causal mask to the entire encoder output
            for mod in self.layers:
                output = mod(output, memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
                
        else:
            output = memory
            # Apply the fixed causal mask to the entire encoder output
            for mod in self.layers:
                output = mod(output, memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
        
        output = self.output_linear(output)
        return output
    
    def forward(self, tgt, intentions, segmentwise=False):
        # Create a fixed causal mask
        mask = self.causal_mask[:intentions.size(0), :intentions.size(0)].to(intentions.device)
        if segmentwise:
            return self.forward_segment(intentions)
        else:
            return self.forward_plain(tgt, intentions,
                                      with_tgt=True,
                                      tgt_mask=mask, memory_mask=mask)


class IntentionTransformer(nn.Module):
    def __init__(self, input_dim, n_heads, num_layers, d_model, d_ff, num_intentions, vocab_size, vocab_size_skill):
        super(IntentionTransformer, self).__init__()
        self.state_embedding = StateEmbedding(13, 13, d_model)
        self.encoder = DirectEncoder(input_dim, n_heads, num_layers, d_model, d_ff, vocab_size_skill)
        self.intention_abstraction = IntentionAbstraction(num_intentions, d_model)
        self.decoder_layer = CausalDecoderLayer(d_model, n_heads)
        self.decoder = MaskedDecoder(self.decoder_layer, num_layers, d_model, vocab_size)
        
        # Store parameters for saving
        self.config = {
            'input_dim': input_dim,
            'n_heads': n_heads,
            'num_layers': num_layers,
            'd_model': d_model,
            'd_ff': d_ff,
            'num_intentions': num_intentions,
            'vocab_size': vocab_size,
            'vocab_size_skill': vocab_size_skill
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
    
    def prepending(self, tokens):
        BOS_token_index = 8 # 8 for <bos> token in addition to 7 minigrid actions
        batch_size = tokens.shape[0]
        bos_prepending = torch.full((batch_size, 1), BOS_token_index, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([bos_prepending, tokens[:, :-1]], dim=-1)
        
        
    def forward(self, skills, states, actions):
        B, T, C, H, W = states.size()
        states = states.view(B * T, C, H, W)
        state_embedding = self.state_embedding(states)
        state_embedding = state_embedding.view(B, T, -1)
        encoder_output, skill_embedding = self.encoder(skills, state_embedding)
        reorganized_intentions, internal_index, mask = self.intention_abstraction(skills, encoder_output, skill_embedding)
        prepended_actions = self.prepending(actions)
        # TODO: Prepending is needed for decoder input.
        actions = self.decoder(prepended_actions, reorganized_intentions)
        return actions, internal_index


if __name__ == "__main__":
    # Example usage
    model = IntentionTransformer(input_dim=128,
                                 n_heads=8,
                                 num_layers=6,
                                 d_model=512,
                                 d_ff=64,
                                 num_intentions=11,
                                 vocab_size=11,
                                 vocab_size_skill=11)
    
    input_actions = torch.randint(0, 10, (32, 20))  # (batch size, time steps)
    input_states = torch.rand(32, 20, 3, 11, 6)  # (batch size * time steps, map_height, map_width, 3)
    output_actions = model(input_actions, input_states)