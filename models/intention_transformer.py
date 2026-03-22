import os
import json
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder
import torch.nn.functional as F
from models.intention_abstraction import IntentionAbstraction
from models.state_embedding import StateEmbedding
from models.causal_decoding_layer import CausalDecoderLayer
from models.ngram_transition import TransitionModel

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
    def __init__(self, d_model, num_intentions, hidden_size=256, num_layers=3, dropout_prob=0.1):
        super(IntentionHead, self).__init__()
        layers = []
        input_size = d_model
        
        # Create multiple hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())  # Alternative activation
            layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size
            hidden_size = max(hidden_size // 2, num_intentions * 2)  # Gradually reduce size
        
        # Final output layer
        layers.append(nn.Linear(input_size, num_intentions))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.mlp(x)
        onehot = F.gumbel_softmax(logits, tau=1.0, hard=True)
        # Argmax to get indices, and add 1 to spare 0 for padding token
        index = onehot.argmax(dim=-1) + 1  # add 1 to spare 0 for padding token
        
        return onehot, index, logits



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
        # Permute the input to make time as the first dimension
        intentions = intentions.permute((1, 0, 2))
        
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
        
        self.num_intentions = num_intentions
        
        self.state_embedding = StateEmbedding(13, 13, d_model)
        self.encoder = TransformerEncoder(input_dim, n_heads, num_layers, d_model, d_ff, vocab_size)
        self.intention_head = IntentionHead(d_model, num_intentions)
        self.intention_abstraction = IntentionAbstraction(num_intentions, d_model)
        self.transition = TransitionModel(d_model, num_intentions, max_length=30)
        self.decoder_layer = CausalDecoderLayer(d_model, n_heads)
        self.decoder = MaskedDecoder(self.decoder_layer,num_layers, d_model, vocab_size)
        
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
    
    def forward(self, actions, states, skills):
        B, T, H, W, C = states.size()
        states = states.view(B * T, H, W, C)
        
        # Embed the state and action
        state_embedding = self.state_embedding(states)
        state_embedding = state_embedding.view(B, T, -1)
        
        # Get representation from tranformer encoder
        encoder_output = self.encoder(actions, state_embedding)
        
        # Translate the input into one-hot intention with the same length
        onehot, index, gumbel_logits = self.intention_head(encoder_output)
        
        # Reorganaze the intention embedding by repeating skill embedding of repeated consecutive skills
        # Extract the distinct skill sequence for transition learning
        reorganized_intentions, mask, intention_embedding_corpus, intention_index_corpus \
            = self.intention_abstraction(onehot, index, encoder_output) # encoder_output is replaced with state_embedding to eliminate temporal info
        
        # Transition model
        transition_logits,  intention_label_corpus = self.transition(intention_embedding_corpus, intention_index_corpus)
        
        # decode the reorganized representation
        recons = self.decoder(reorganized_intentions)
        return recons, onehot, gumbel_logits, transition_logits, intention_label_corpus


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