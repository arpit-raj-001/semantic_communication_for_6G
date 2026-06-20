import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.nn.functional as F
from rouge_score import rouge_scorer

class SemanticMetrics:
   
    def __init__(self, tokenizer):
        self.smooth = SmoothingFunction().method1
        self.tokenizer = tokenizer
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_bleu(self, reference_tokens: list, candidate_tokens: list) -> float:
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smooth)
        
    def calculate_rouge(self, reference_text: str, candidate_text: str) -> float:
        scores = self.rouge.score(reference_text, candidate_text)
        return scores['rougeL'].fmeasure

    def calculate_ber_proxy(self, reference_tokens: list, candidate_tokens: list) -> float:
        if len(reference_tokens) == 0: return 1.0
        #pad if short
        padded_cand = candidate_tokens + [0]*max(0, len(reference_tokens) - len(candidate_tokens))
        errors = sum(1 for r, c in zip(reference_tokens, padded_cand) if r != c)
        return min(errors / len(reference_tokens), 1.0)

    def calculate_cosine_similarity(self, model, ref_tokens_tensor, cand_tokens_tensor) -> float:
        with torch.no_grad():
            _, (h_ref, _) = model.encoder.lstm(model.encoder.embedding(ref_tokens_tensor))
            _, (h_cand, _) = model.encoder.lstm(model.encoder.embedding(cand_tokens_tensor))
            sim = F.cosine_similarity(h_ref, h_cand, dim=-1).item()
            return max(sim, 0.0)

    def evaluate_batch(self, model, source_tokens, target_tokens, snr_db, pad_id=0):
        
        model.eval()
        with torch.no_grad():
            outputs = model(source_tokens, target_tokens, snr_override=snr_db)
            predictions = torch.argmax(outputs, dim=-1)
            
            batch_size = source_tokens.size(0)
            t_bleu, t_rouge, t_ber, t_cos = 0.0, 0.0, 0.0, 0.0
            
            for i in range(batch_size):
                ref = target_tokens[i].tolist()
                cand = predictions[i].tolist()
                
                ref_clean = [t for t in ref if t != pad_id]
                cand_clean = [t for t in cand if t != pad_id]
                ref_text = self.tokenizer.decode(ref_clean)
                cand_text = self.tokenizer.decode(cand_clean)
                
                # 1. BLEU
                t_bleu += self.calculate_bleu(ref_clean, cand_clean)
                # 2. ROUGE
                t_rouge += self.calculate_rouge(ref_text, cand_text)
                # 3. BER
                t_ber += self.calculate_ber_proxy(ref_clean, cand_clean)
                # 4. Cosine Similarity
                t_cos += self.calculate_cosine_similarity(model, target_tokens[i].unsqueeze(0), predictions[i].unsqueeze(0))
                
            return (t_bleu / batch_size, 
                    t_rouge / batch_size, 
                    t_ber / batch_size, 
                    t_cos / batch_size)
