# 1.2 Tokenization Theory:

Before we can compress a sentence into a continuous mathematical vector, we must convert the raw text into discrete integers that the neural network can process. This process is called **Tokenization**.

### The Lookup table

The neural network cannot read strings. We must scan our `EuroparlDataLoader` dataset to build a lookup table mapping every unique word to a specific integer ID (e.g., `"Arpit" -> 1242`).

### Special Tokens

To train an LSTM Seq2Seq model over a noisy channel, there are 4 special tokens

1. **`<PAD>` (Padding - ID: 0):**
   If Sentence A has 5 words and Sentence B has 10 words, we must append five `<PAD>` tokens to Sentence A so both are length 10. because neural network process in batches

2. **`<UNK>` (Unknown - ID: 1):**
   the tokenizer maps unseen words to `<UNK>`. This forces the LSTM to rely on surrounding context to guess the meaning of the unknown word

3. **`<BOS>` (Begin Of Sequence - ID: 2):**
   when thee neural network is decoding the output , it needs a sstart point to know its time to start predicting. We feed it `<BOS>`.

4. **`<EOS>` (End Of Sequence - ID: 3):**
   The LSTM Decoder will keep predicting words forever unless it generates an `<EOS>` token. Once `<EOS>` is predicted, the receiver knows the semantic thought is complete.

### Word-Level vs. Subword Tokenization

Modern LLMs (like the T5 model we used in the sandbox) use _Subword Tokenization_ (Byte-Pair Encoding) , they chop 'aadya' into 'aad' , 'ya'
