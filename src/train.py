import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\Shrish\Desktop\semantic-comm\actual_project')))

from src.data_loader import EuroparlDataLoader
from src.tokenizer import SemanticTokenizer
from src.model import JointSemanticModel

class SemanticDataset(Dataset):
    
    def __init__(self, sentences, tokenizer, max_len=20):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        #string to ids
        encoded = self.tokenizer.encode(self.sentences[idx], max_length=self.max_len)
        return torch.tensor(encoded, dtype=torch.long)


def train_model():
    print("training..")
    
    #data load
    data_dir = r"C:\Users\Shrish\Desktop\semantic-comm\Semantic_Communication\europarl\en\en"
    loader = EuroparlDataLoader(data_dir=data_dir, target_sentences=1000)
    loader.scan_and_load()
    
    if len(loader.all_sentences) == 0:
        print("Europarl data not found. Using a dummy dataset instead")
        sentences = [
            "The weather is nice today.", 
            "i lost my mind and no one noticed",
            "i dont know why i like you so much , literally publishing a research paper for you"
        ] * 333
    else:
        sentences = loader.all_sentences
        
    #tokenization
    tokenizer = SemanticTokenizer(min_freq=2)
    tokenizer.fit(sentences)
    max_sequence_length = 30
    dataset = SemanticDataset(sentences, tokenizer, max_len=max_sequence_length)
    
    
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    
    #train with jscc at 10snr
    model = JointSemanticModel(
        vocab_size=tokenizer.vocab_size, 
        embed_dim=128, 
        hidden_dim=256, 
        snr_db=10.0
    )
    #loss function and optimation
   
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #ignore pads
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID)
    
    epochs = 3
    print(f"\nTraining on {len(sentences)} sentences for {epochs} epochs...")
    print(f"Vocab Size: {tokenizer.vocab_size} \n")
    
    #training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            source_tokens = batch
            target_tokens = batch
            
            # Forward pass: Encoder -> Physical AWGN Channel -> Decoder
            # Outputs shape: [batch_size, sequence_length, vocab_size]
            outputs = model(source_tokens, target_tokens)
            
            #ignore bos and flatten to calculate entropy loss
            outputs_flat = outputs[:, :-1, :].reshape(-1, tokenizer.vocab_size)
            targets_flat = target_tokens[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            
            #update weightd
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx:02d} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")
        
    print("\ndone training!")
    
    
if __name__ == "__main__":
    train_model()
