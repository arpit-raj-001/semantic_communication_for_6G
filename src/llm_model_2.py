import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.channel import AWGNChannel

class AdvancedGenerativeSemanticModel(nn.Module):
    """
    Phase 2 Advanced (Step 2.2): Channel Adaptation via SNR Embedding.
    This model mathematically ingests the Channel Noise (SNR) and adapts
    its semantic manifold to protect information dynamically.
    """
    def __init__(self, model_name="t5-small", snr_db=10.0):
        super().__init__()
       
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.channel = AWGNChannel(snr_db=snr_db)
        
        # Adaptive SNR Embedding Layer (Projects 1D SNR value to T5 Hidden Dimension)
        d_model = self.t5.config.d_model
        self.snr_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, d_model)
        )

    def _get_snr_tensor(self, batch_size, device, snr_override):
        if snr_override is None:
            snr_val = self.channel.snr_db
            return torch.full((batch_size, 1), snr_val, dtype=torch.float32, device=device)
        elif isinstance(snr_override, (float, int)):
            return torch.full((batch_size, 1), float(snr_override), dtype=torch.float32, device=device)
        elif isinstance(snr_override, torch.Tensor):
            return snr_override.view(-1, 1).to(dtype=torch.float32, device=device)
        else:
            raise ValueError("snr_override must be None, float, or Tensor")

    def forward(self, input_ids, attention_mask=None, labels=None, snr_override=None):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pristine_hidden_states = encoder_outputs.last_hidden_state
        
        # 1. Embed the SNR
        snr_tensor = self._get_snr_tensor(batch_size, device, snr_override)
        snr_embedding = self.snr_embed(snr_tensor).unsqueeze(1) # [Batch, 1, d_model]
        
        # 2. Add SNR Embedding to encoder outputs so the Decoder knows the channel state
        adapted_hidden_states = pristine_hidden_states + snr_embedding
        
        # 3. Pass through physical noise
        noisy_hidden_states = self.channel(adapted_hidden_states, snr_db_override=snr_override)
        
        encoder_outputs.last_hidden_state = noisy_hidden_states

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=True
        )
        return outputs
        
    def generate(self, input_ids, attention_mask=None, snr_override=None, max_length=20):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pristine_hidden_states = encoder_outputs.last_hidden_state
        
        snr_tensor = self._get_snr_tensor(batch_size, device, snr_override)
        snr_embedding = self.snr_embed(snr_tensor).unsqueeze(1)
        adapted_hidden_states = pristine_hidden_states + snr_embedding
        
        noisy_hidden_states = self.channel(adapted_hidden_states, snr_db_override=snr_override)
        encoder_outputs.last_hidden_state = noisy_hidden_states
        
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length
        )

if __name__ == "__main__":
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = AdvancedGenerativeSemanticModel(snr_db=5.0)
    text = "Semantic communication is the future."
    inputs = tokenizer(text, return_tensors="pt")
    
    labels = inputs["input_ids"].clone()
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels, snr_override=-5.0)
    print(f"Adaptive Forward Pass Loss at -5dB SNR: {outputs.loss.item():.4f}")
