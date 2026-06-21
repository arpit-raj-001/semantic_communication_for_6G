import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\Shrish\Desktop\semantic-comm\actual_project')))
from src.data_loader import EuroparlDataLoader
from src.llm_model import GenerativeSemanticModel

class EuroparlT5Dataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=20):
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
        #to ignore pad as labels we need it to be -100 in cross entropy
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return input_ids, attention_mask, labels


def train_llm_jscc():
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    
    data_dir = r"C:\Users\Shrish\Desktop\semantic-comm\Semantic_Communication\europarl\en\en"
    loader = EuroparlDataLoader(data_dir=data_dir, target_sentences=150000)
    loader.scan_and_load()
    sentences = loader.all_sentences if len(loader.all_sentences) > 0 else ["The dataset was not found , this sentence is a dummy to prevent crash"] * 50000
    
    print(f"Total sentences loaded: {len(sentences)}")
    
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = EuroparlT5Dataset(sentences, tokenizer, max_length=20)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = GenerativeSemanticModel(model_name="t5-small", snr_db=5.0).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 3
    print(f"\n{epochs} Epochs")
    print(f"Total Batches per Epoch: {len(dataloader)}")
    
    
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} ")
        total_loss = 0.0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx:04d}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} \n")
        
    print("Done!")
    
    
    models_dir = os.path.abspath(r"C:\Users\Shrish\Desktop\semantic-comm\actual_project\models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "t5_semantic.pt")
    torch.save(model.state_dict(), model_path)
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train_llm_jscc()
