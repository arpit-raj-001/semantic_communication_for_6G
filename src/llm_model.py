import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.channel import AWGNChannel
#based on t5
class GenerativeSemanticModel(nn.Module):
    
    def __init__(self, model_name="t5-small", snr_db=10.0):
        super().__init__()
       
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.channel = AWGNChannel(snr_db=snr_db)

    def forward(self, input_ids, attention_mask=None, labels=None, snr_override=None):
        
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pristine_hidden_states = encoder_outputs.last_hidden_state
        noisy_hidden_states = self.channel(pristine_hidden_states, snr_db_override=snr_override)
        
        #noisy vector goes to llm in hugging face format
        encoder_outputs.last_hidden_state = noisy_hidden_states

    
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=True
        )
        return outputs
        
    def generate(self, input_ids, attention_mask=None, snr_override=None, max_length=20):
        
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        noisy_hidden = self.channel(encoder_outputs.last_hidden_state, snr_db_override=snr_override)
        encoder_outputs.last_hidden_state = noisy_hidden
        
        
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length
        )

if __name__ == "__main__":
    
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = GenerativeSemanticModel(snr_db=5.0)
    
    text = "Semantic communication is the future."
    inputs = tokenizer(text, return_tensors="pt")
    
    
    labels = inputs["input_ids"].clone()
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
    print(f"Forward Pass Loss at 5dB SNR: {outputs.loss.item():.4f}")
    gen_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    print(f"Untrained Decoding (Hallucination): {tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")
