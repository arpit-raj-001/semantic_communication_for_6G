import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

class EuroparlDataLoader:
    """
    Handles the loading, cleaning, and splitting of the Europarl knowleedge base
    """
    
    def __init__(self, data_dir: str, target_sentences: int = 40000, length_threshold: int = 30):
        """
        Initialize the data loader.
            data_dir: Path to the Europarl text files
            target_sentences: max number of sentences to load
            length_threshold: small vs big sentence
        """
        self.data_dir = Path(data_dir)
        self.target_sentences = target_sentences
        self.length_threshold = length_threshold
        self.all_sentences: List[str] = []
        self.small_sentences: List[str] = []
        self.big_sentences: List[str] = []
        # load karne waqt ye empty arrays bharenge , abhi data nhi mere paas (WIP)

    def _is_english_like(self, text: str) -> bool:

        # a function to check if text is english or not , cause if not it might be xml tags or corrupred data , we dont want to train on them , it will destroy our semantic space
        
        if len(text.split()) < 3:
            return False
            
        # Must contain at least one vowel
        if not re.search(r"[aeiouAEIOU]", text):
            return False
            
        #ensure ascii text
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        if (ascii_chars / max(len(text), 1)) < 0.95:
            return False
            
        return True

    def scan_and_load(self) -> None:
       
        if not self.data_dir.exists():
            raise FileNotFoundError(f"europarl dataset not found {self.data_dir} ")

        files = sorted(self.data_dir.rglob('*.txt'))
        if not files:
            raise FileNotFoundError(f"no .txt files found in {self.data_dir}")

        total_loaded = 0
        
        for file_path in files:
           
            if total_loaded >= self.target_sentences:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        stripped = line.strip()
                        
                        # Skip empty lines and XML markup lines like <>
                        if not stripped or (stripped.startswith('<') and stripped.endswith('>')):
                            continue
                            
                        if self._is_english_like(stripped):
                            self.all_sentences.append(stripped)
                            total_loaded += 1
                            
                            if total_loaded >= self.target_sentences:
                                break
            except Exception as e:
                print(f"skipping file {file_path} due to error: {e}")
                continue

        #deduplicate dataset
        unique_sentences = list(dict.fromkeys(self.all_sentences))
        self.all_sentences = unique_sentences

    def categorize_by_length(self) -> None:
        # divide dataset based on word count , we will see effects on long sentence vs short sentence

        for sentence in self.all_sentences:
            word_count = len(sentence.split())
            if word_count <= self.length_threshold:
                self.small_sentences.append(sentence)
            else:
                self.big_sentences.append(sentence)

    def get_splits(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42) -> Dict[str, List[str]]:
        # split into train,validation,test dta set
        def split_data(sentences: List[str]) -> Tuple[List[str], List[str], List[str]]:
            if not sentences:
                return [], [], []
            #train+val vs test
            train_val, test = train_test_split(sentences, test_size=test_size, random_state=random_state)
            #train vs val
            val_ratio = val_size / (1 - test_size)
            train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)
            return train, val, test

        small_train, small_val, small_test = split_data(self.small_sentences)
        big_train, big_val, big_test = split_data(self.big_sentences)
        train_set = small_train + big_train
        val_set = small_val + big_val
        test_set = small_test + big_test

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }

if __name__ == "__main__":
    
    print("wait loading...")
    loader = EuroparlDataLoader(data_dir=r"C:\Users\Shrish\Desktop\semantic-comm\actual_project\europarl\en\en")
    loader.scan_and_load()
    loader.categorize_by_length()
    splits = loader.get_splits()
    
    print(f"Total Sentences Loaded (Unique): {len(loader.all_sentences)}")
    print(f"Small Sentences: {len(loader.small_sentences)}")
    print(f"Big Sentences: {len(loader.big_sentences)}")
    print(f"Train/Val/Test Split: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
