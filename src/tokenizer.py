import re
from typing import List, Dict, Optional
from collections import Counter

class SemanticTokenizer:
    # english to integer
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, min_freq: int = 2):
        # min freq tells atleast kitni baar ye word ho dataaset me to be counted as a word , otherwise treat as UNK , this is done to reduce typos 
        self.min_freq = min_freq
        
        
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID
        }
        
        #making lookup table
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        
        self.vocab_size = len(self.word2idx)
        self.is_fit = False

    def _clean_and_split(self, sentence: str) -> List[str]:
        # convert to lowercase and isolate punctuation or question marks , because they carry heavy intent

        sentence = sentence.lower()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        return sentence.strip().split()

    def fit(self, sentences: List[str]) -> None:
        # dataset to vocabulary dictionary
        print("Scanning dataset to build vocabulary...")
        word_counts = Counter()
        
        for sentence in sentences:
            words = self._clean_and_split(sentence)
            word_counts.update(words)
            
        
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
        self.is_fit = True
        print(f"dictionary size: {self.vocab_size}")

    def encode(self, sentence: str, max_length: Optional[int] = None) -> List[int]:
    #    raw string to list of integer and wrap around eos and bos
        if not self.is_fit:
            raise RuntimeError("Tokenizer must be fit on a dataset before encoding.")
            
        words = self._clean_and_split(sentence)
        
        
        encoded = [self.BOS_ID]
        for word in words:
            encoded.append(self.word2idx.get(word, self.UNK_ID))
            
        encoded.append(self.EOS_ID)
        
        #pad/truncate to meet fixed size
        if max_length is not None:
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
                encoded[-1] = self.EOS_ID
            else:
                padding_needed = max_length - len(encoded)
                encoded.extend([self.PAD_ID] * padding_needed)
                
        return encoded

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        
        words = []
        for idx in token_ids:
            if idx == self.EOS_ID and skip_special_tokens:
                break
                
                # skip padd and bos while reading
            if skip_special_tokens and idx in [self.PAD_ID, self.BOS_ID]:
                continue
                
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
            
        decoded_text = " ".join(words)
        decoded_text = re.sub(r" ([?.!,¿])", r"\1", decoded_text)
        return decoded_text

if __name__ == "__main__":
    #a demo of tokenization
    toy_dataset = [
        "Global warming is a big issue",
        "how do i tell her i like her for who she is",
        "i want to publish a research paper"
    ]
    
    tokenizer = SemanticTokenizer(min_freq=1)
    tokenizer.fit(toy_dataset)
    
    test_sentence = "i have lost my mind for her"
    print(f"\nOriginal Sentence: '{test_sentence}'")
    
    encoded_ids = tokenizer.encode(test_sentence, max_length=50)
    print(f"Transmitted IDs:   {encoded_ids}")
    
    decoded_sentence = tokenizer.decode(encoded_ids)
    print(f"Decoded Sentence:  '{decoded_sentence}'")
    
    
