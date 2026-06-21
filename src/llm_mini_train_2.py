import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
import torch
import sys
import os

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

def mini_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
   
    data_dir = r"C:\Users\Shrish\Desktop\semantic-comm\Semantic_Communication\europarl\en\en"
    loader = EuroparlDataLoader(data_dir=data_dir, target_sentences=9000)
    loader.scan_and_load()
    sentences = loader.all_sentences if len(loader.all_sentences) > 0 else ["Dummy dataset protection"] * 5000
    
    print(f"Total sentences : {len(sentences)}")
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset = EuroparlT5Dataset(sentences, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = AdvancedGenerativeSemanticModel(model_name="t5-small", snr_db=10.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    
    epochs = 3
    
    
    model.train()
    for epoch in range(epochs):
        if epoch == 0:
            print(f"Clean: (Epoch {epoch+1}/{epochs}) ")
            current_stage = "Clean"
        else:
            print(f"Adaptive (epoch) {epoch+1}/{epochs}) ---")
            current_stage = "Adaptive"
            
        total_loss = 0.0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if current_stage == "Clean":
                batch_snr = 20.0
            else:
                import random
                batch_snr = random.uniform(-10.0, 20.0)
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, snr_override=batch_snr)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx:04d}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}  | Average Loss: {avg_loss:.4f} ===\n")
        
    print("done")
    
    models_dir = os.path.abspath(r"C:\Users\Shrish\Desktop\semantic-comm\actual_project\models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "t5_semantic_advanced.pt")
    
    torch.save(model.state_dict(), model_path)
    print(f"\nMini-Trained Model saved to: {model_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    mini_train()
