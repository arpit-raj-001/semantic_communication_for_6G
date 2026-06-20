import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

class EuroparlDataLoader:
    """
    Handles the loading, cleaning, and splitting of the Europarl corpus.
    This acts as the 'World Knowledge' provider for our Semantic Communication system.
    """
    
    def __init__(self, data_dir: str, target_sentences: int = 20000, length_threshold: int = 30):
        """
        Initializes the data loader.
        
        Args:
            data_dir: Path to the Europarl text files (e.g., 'europarl/en/en').
            target_sentences: Maximum number of valid sentences to load to prevent memory overload.
            length_threshold: Word count boundary separating 'Small' vs 'Big' sentences.
        """
        self.data_dir = Path(data_dir)
        self.target_sentences = target_sentences
        self.length_threshold = length_threshold
        
        # We will populate these arrays during the load process
        self.all_sentences: List[str] = []
        self.small_sentences: List[str] = []
        self.big_sentences: List[str] = []

    def _is_english_like(self, text: str) -> bool:
        """
        Heuristic check to ensure a line of text is actually a valid English sentence.
        Raw datasets often contain XML tags, speaker names, or corrupted characters.
        
        Why this matters in Telecommunications: 
        If our Neural Network learns from corrupted XML tags instead of human language, 
        its 'semantic space' will be misaligned, leading to decoding failures later.
        """
        # Sentences shorter than 3 words usually lack semantic depth (e.g., "Yes.")
        if len(text.split()) < 3:
            return False
            
        # Must contain at least one vowel
        if not re.search(r"[aeiouAEIOU]", text):
            return False
            
        # Ensure the text is primarily ASCII to avoid weird encoding artifacts
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        if (ascii_chars / max(len(text), 1)) < 0.95:
            return False
            
        return True

    def scan_and_load(self) -> None:
        """
        Scans the directory for .txt files and extracts clean English sentences.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found. Please ensure Europarl dataset is present.")

        files = sorted(self.data_dir.rglob('*.txt'))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")

        total_loaded = 0
        
        for file_path in files:
            # Stop if we hit our target limit (saving RAM and compute time)
            if total_loaded >= self.target_sentences:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        stripped = line.strip()
                        
                        # Skip empty lines and XML markup like <SPEAKER 1>
                        if not stripped or (stripped.startswith('<') and stripped.endswith('>')):
                            continue
                            
                        # If it passes the heuristic, it's a valid semantic datapoint
                        if self._is_english_like(stripped):
                            self.all_sentences.append(stripped)
                            total_loaded += 1
                            
                            if total_loaded >= self.target_sentences:
                                break
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
                continue

        # Deduplicate the dataset to prevent data leakage between train/val/test splits
        unique_sentences = list(dict.fromkeys(self.all_sentences))
        self.all_sentences = unique_sentences

    def categorize_by_length(self) -> None:
        """
        Divides the dataset into Small and Big sentences based on word count.
        
        Why this matters:
        We will later test our Neural Network's ability to compress long thoughts vs short thoughts.
        A 384-dimensional vector can easily store a 5-word sentence, but a 100-word sentence 
        experiences heavy 'semantic compression'.
        """
        for sentence in self.all_sentences:
            word_count = len(sentence.split())
            if word_count <= self.length_threshold:
                self.small_sentences.append(sentence)
            else:
                self.big_sentences.append(sentence)

    def get_splits(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42) -> Dict[str, List[str]]:
        """
        Splits the categorized data into Train, Validation, and Test sets.
        
        Returns a dictionary containing the splits.
        """
        def split_data(sentences: List[str]) -> Tuple[List[str], List[str], List[str]]:
            if not sentences:
                return [], [], []
            # First split: Train+Val vs Test
            train_val, test = train_test_split(sentences, test_size=test_size, random_state=random_state)
            # Second split: Train vs Val
            val_ratio = val_size / (1 - test_size)
            train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)
            return train, val, test

        small_train, small_val, small_test = split_data(self.small_sentences)
        big_train, big_val, big_test = split_data(self.big_sentences)

        # Combine them back together
        train_set = small_train + big_train
        val_set = small_val + big_val
        test_set = small_test + big_test

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }

if __name__ == "__main__":
    # Example usage for verification
    # Using the standard path mentioned in the research repo
    print("Initializing Europarl DataLoader...")
    loader = EuroparlDataLoader(data_dir=r"C:\Users\Shrish\Desktop\semantic-comm\Semantic_Communication\europarl\en\en")
    loader.scan_and_load()
    loader.categorize_by_length()
    splits = loader.get_splits()
    
    print(f"Total Sentences Loaded (Unique): {len(loader.all_sentences)}")
    print(f"Small Sentences: {len(loader.small_sentences)}")
    print(f"Big Sentences: {len(loader.big_sentences)}")
    print(f"Train/Val/Test Split: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
