import torch
import torch.nn as nn

class AWGNChannel(nn.Module):
    def __init__(self, snr_db: float = 10.0):
        super(AWGNChannel, self).__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor, snr_db_override: float = None) -> torch.Tensor:
        """
            x: The transmitted signal vector.
            snr_db_override: dynamic SNR for testing different channel conditions            
        gives:
            The noisy received signal vector.
        """
        current_snr = snr_db_override if snr_db_override is not None else self.snr_db
        if current_snr >= 100.0:
            return x

        # power of signal
        signal_power = torch.mean(x ** 2)
        snr_linear = 10 ** (current_snr / 10.0)

        #P_noise = P_signal / SNR
        noise_power = signal_power / snr_linear

        
        # torch.randn generates standard normal noise N(0, 1)
        # scale by standard deviation (which is the square root of variance/power)
        noise_std_dev = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * noise_std_dev
        received_signal = x + noise
        
        return received_signal

if __name__ == "__main__":
    
    torch.manual_seed(42)
    
    # Simulate a batch of 2 sentences, each compressed into a 5-dimensional vector
    dummy_signal = torch.ones(2, 5) * 2.0  #amplitude 2
    print("Transmitted Signal")
    print(dummy_signal)
    
    channel = AWGNChannel(snr_db=10.0)
    received_good = channel(dummy_signal)
    print("\n recieved when snr is good ")
    print(received_good)
    
    received_bad = channel(dummy_signal, snr_db_override=-5.0)
    print("\nrecieved when snr is bad")
    print(received_bad)
    
