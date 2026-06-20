# 1.3 AWGN Channel Physics

In traditional deep learning (like Computer Vision or NLP), we assume that the mathematical vectors passed between layers are transferred perfectly.

In Semantic Communication, we simulate transmitting our latent vector across a physical wireless environment . To do this, we insert a specialized neural layer directly in the middle of our network: the **AWGN Channel**.

### What is AWGN?

AWGN stands for **Additive White Gaussian Noise**.

- **Additive:** The noise is added directly to the signal ($Y = X + N$).
- **White:** The noise contains equal power across all frequencies (like white light).
- **Gaussian:** The magnitude of the noise follows a normal (Gaussian) bell-curve distribution.

### Understanding SNR (Signal-to-Noise Ratio)

We don't measure channel quality by absolute noise; we measure it by the _ratio_ of our signal's power compared to the noise's power. This is measured in **Decibels (dB)**.

The formula is:
$$ \text{SNR (dB)} = 10 \cdot \log*{10} \left( \frac{P*{\text{signal}}}{P\_{\text{noise}}} \right) $$

**What the numbers mean in reality:**

- **20 dB:** Excellent connection. Signal is 100x stronger than noise.
- **10 dB:** Good connection. Signal is 10x stronger than noise.
- **0 dB:** Critical boundary. Signal and noise are equally loud.
- **-5 dB:** Severe degradation. The noise is more than 3x louder than the actual signal!

### The Mathematical Bottleneck

We must dynamically calculate the average power of the transmitted batch of vectors ($P_{\text{signal}}$), convert our target SNR from Decibels back to a linear scale, calculate the required $P_{\text{noise}}$, and then inject `torch.randn` scaled by $\sqrt{P_{\text{noise}}}$.

This layer makes our Neural Network non-deterministic. The Decoder will never see the exact same mathematical vector twice, forcing it to learn robust, generalized semantic meaning rather than memorizing exact floating-point values.
