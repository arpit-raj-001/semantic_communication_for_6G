import torch
import torch.nn as nn
from typing import Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.channel import AWGNChannel

class SemanticEncoder(nn.Module):
    #read all word id and then add their meaning into a single vector
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super(SemanticEncoder, self).__init__()
        #id to vector
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        #process the sequence and build memory
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # x is batch of tokenized sentences
        # it returns hidden states and cell states 
    
        embedded = self.embedding(x)
        
        
        _, (hidden, cell) = self.lstm(embedded)
        
        return hidden, cell


class SemanticDecoder(nn.Module):
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super(SemanticDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_step: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        """
            input_step: The PREVIOUSLY predicted word or BOS
            hidden: The current noisy hidden state from the channel/previous step.
            cell: The current noisy cell state from the channel/previous step.
        Returns:
            next_hidden, next_cell
        """
        
        embedded = self.embedding(input_step)
        output, (next_hidden, next_cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, next_hidden, next_cell


class JointSemanticModel(nn.Module):
   #jscc pipeline
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, snr_db: float = 10.0):
        super(JointSemanticModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.encoder = SemanticEncoder(vocab_size, embed_dim, hidden_dim)
        self.channel = AWGNChannel(snr_db=snr_db)
        self.decoder = SemanticDecoder(vocab_size, embed_dim, hidden_dim)

    def forward(self, source_tokens: torch.Tensor, target_tokens: torch.Tensor, snr_override: float = None):
        """
        Runs the full transmission pipeline (used during Training).
        
        Args:
            source_tokens: The text to transmit. Shape: [batch_size, seq_length]
            target_tokens: The ground truth text used for "Teacher Forcing". Shape: [batch_size, seq_length]
            snr_override: Dynamic channel condition testing.
        """
        batch_size = source_tokens.size(0)
        target_len = target_tokens.size(1)
        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(source_tokens.device)
        hidden, cell = self.encoder(source_tokens)
        noisy_hidden = self.channel(hidden, snr_db_override=snr_override)
        noisy_cell = self.channel(cell, snr_db_override=snr_override)
        
        
        input_step = target_tokens[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            
            prediction, noisy_hidden, noisy_cell = self.decoder(input_step, noisy_hidden, noisy_cell)
            outputs[:, t, :] = prediction
            input_step = target_tokens[:, t].unsqueeze(1)
            
        return outputs

if __name__ == "__main__":
    vocab_size = 500
    model = JointSemanticModel(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, snr_db=10.0)
    
    # Simulate a batch of 2 sentences, length 10
    dummy_source = torch.randint(1, vocab_size, (2, 10))
    dummy_target = torch.randint(1, vocab_size, (2, 10))
    
    output_logits = model(dummy_source, dummy_target)
    print(f"joint model forward pass done")
    print(f"Output Tensor Shape: {output_logits.shape} (Expected: [2, 10, {vocab_size}])")
