#this one is channel aware model much better than model1


# import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\Shrish\Desktop\semantic-comm\actual_project')))
from src.data_loader import EuroparlDataLoader
from src.llm_model_2 import AdvancedGenerativeSemanticModel

class EuroparlT5Dataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=32):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return input_ids, attention_mask, labels


def train_curriculum_jscc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    # We use a large dataset to ensure the model generalizes the manifold
    data_dir = r"C:\Users\Shrish\Desktop\semantic-comm\Semantic_Communication\europarl\en\en"
    loader = EuroparlDataLoader(data_dir=data_dir, target_sentences=150000)
    loader.scan_and_load()
    sentences = loader.all_sentences if len(loader.all_sentences) > 0 else ["Dummy dataset protection"] * 50000
    
    print(f"Total sentences loaded: {len(sentences)}")
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = EuroparlT5Dataset(sentences, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Load the Advanced Model with the SNR Embedding Layer
    model = AdvancedGenerativeSemanticModel(model_name="t5-small", snr_db=10.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 5
    print(f"\nCommencing Phase 2 Advanced Curriculum Training...")
    
    model.train()
    for epoch in range(epochs):
        # Stage 1: Clean Pretraining (Epoch 1)
        # Stage 2: Adaptive JSCC (Epoch 2+)
        if epoch == 0:
            print(f"--- Stage 1: Clean Channel Pretraining (Epoch {epoch+1}/{epochs}) ---")
            current_stage = "Clean"
        else:
            print(f"--- Stage 2: Adaptive JSCC Curriculum (Epoch {epoch+1}/{epochs}) ---")
            current_stage = "Adaptive"
            
        total_loss = 0.0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if current_stage == "Clean":
                # Override SNR to 20 dB (practically no noise) to learn clean linguistics
                batch_snr = 20.0
            else:
                # Randomly sample SNR between -10 dB and 20 dB for each batch
                # The model's snr_embedding layer dynamically adjusts the manifold
                batch_snr = torch.empty(input_ids.size(0)).uniform_(-10.0, 20.0).to(device)
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, snr_override=batch_snr)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx:04d}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ===\n")
        
    print("Advanced Curriculum Training Phase Complete!")
    
    models_dir = os.path.abspath(r"C:\Users\Shrish\Desktop\semantic-comm\actual_project\models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "t5_semantic_advanced.pt")
    
    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Advanced Adaptive Model saved to: {model_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train_curriculum_jscc()
