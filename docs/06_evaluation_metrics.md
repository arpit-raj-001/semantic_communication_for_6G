# 1.6 Evaluation Metrics

In classical Shannon Telecommunications, the universal metric of success is the **Bit Error Rate (BER)**.
it just calculates how many bits flipped

In Semantic Communication, BER is a mathematically meaningless metric. We are not transmitting bits; we are transmitting high-dimensional vectors. Furthermore, our goal is not syntactic preservation, it is **Intent Preservation**.

### The BLEU Score

To evaluate our system, we borrow from the field of Natural Language Processing (NLP). We utilize the **Bilingual Evaluation Understudy (BLEU)** score.
BLEU measures the $n$-gram precision overlap between the transmitted sentence and the received sentence.

In our repository, we use NLTK's `sentence_bleu`. By evaluating our `JointSemanticModel` across a sliding scale of SNR values (e.g., from 20 dB down to -5 dB), we can plot the "Semantic Waterfall Curve"—the definitive visual proof of a Semantic Communication system's resilience to physical noise.
