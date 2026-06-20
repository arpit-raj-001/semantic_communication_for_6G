# 1.5 Baseline Training Loop:

Training a standard Natural Language Processing (NLP) model is straightforward: you show it a sentence, it predicts the next word, you calculate the loss, and you update the weights.

Training a **Joint Source-Channel Coding (JSCC)** model is significantly more difficult. Because of the `AWGNChannel` sitting in the middle of our network, the neural receiver never sees a clean vector. It has to learn how to actively filter out random Gaussian noise to guess what the transmitter actually meant

### The Physics of JSCC Backpropagation

During training, the PyTorch autograd engine does something remarkable: it backpropagates the gradient _through_ the physical AWGN noise.
This forces the Encoder and Decoder to collaborate.

- The **Encoder** learns to arrange the semantic vectors in a way that maximizes the distance between them, making them less likely to overlap when noise is added.
- The **Decoder** learns to draw complex, non-linear decision boundaries around these noisy clusters to pull the correct word back out.

### The Problem with `<PAD>`

Because our neural network trains on fixed-size matrices, short sentences are filled with `<PAD>` (0) tokens.
If we do not explicitly tell our loss function to ignore the `<PAD>` token, the neural network will spend the vast majority of its time trying to perfectly optimize the transmission of "empty space" over the radio waves, completely destroying its ability to transmit actual semantic words. We handle this using `nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID)`.

### Training at a Fixed SNR

In this baseline phase, we train the model at a fixed SNR (e.g., **10 dB**). This means the model will become a specialist at operating in 10 dB environments. (In Phase 2, we will introduce dynamic SNR training curricula to build a model that can survive fluctuating weather conditions).
