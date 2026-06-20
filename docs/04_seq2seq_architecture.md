# 1.4 Seq2Seq Architecture: The Semantic Bottleneck

modern Attention-based models (like BART) pass an entire matrix of vectors (one for each word) across the network. While powerful, this violates the physics of traditional single-antenna telecommunication which needs fixed size space

To prove the core physics of Semantic Communication, we revert to the classic **LSTM Sequence-to-Sequence (Seq2Seq)** architecture.

### The Encoder (The Neural Transmitter)

The Encoder's job is to read the sentence word-by-word and compress the _entire semantic meaning_ of the sentence into a single mathematical state.

1. It converts discrete word IDs (tokens) into embeddings
2. It passes these vector embedding through an LSTM.
3. It throws away the outputs for every word _except_ the very last one.
4. The final `[hidden, cell]` state of the LSTM becomes the final output layer

### The Decoder (The Neural Receiver)

The Decoder's job is to wake up, grab the mathematical payload out of the air, and translate it back into human language.

1. It initializes its own LSTM's starting memory using the noisy `[hidden, cell]` state received from the channel.
2. It is fed the `<BOS>` (Begin of Sequence) token to trigger the process.
3. It recursively predicts the next word, feeds that prediction back into itself, and continues until it generates an `<EOS>` (End of Sequence) token

### Joint Source-Channel Coding (JSCC)

Normally, engineers build the Transmitter (Source) and the Antenna (Channel) completely separately.
By placing the `AWGNChannel` directly between our PyTorch Encoder and Decoder, we allow the network to perform **Joint Source-Channel Coding**.
