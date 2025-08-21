import torch
import math
import random
import numpy as np
from matplotlib import pyplot as plt

from utils import inspect_dataset, count_sentences, PowerNormalize, \
    list_checkpoints


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def TimeVaryingRician(self, Tx_sig, n_var, M_options=[3, 5, 6, 10], K=1,
                      csi_lag=1):
    """
    Time-varying Rician channel with block-wise fading and lagged equalization.
    Simulates a fast-moving receiver (e.g., 50–150 km/h) where the channel changes
    every block (~2 ms), and equalization uses an outdated channel estimate (lagged
    by csi_lag blocks, default=1 for ~2 ms lag, realistic for 5G with frequent pilots).
    - Tx_sig: Input signal tensor [batch_size, sequence_length, feature_dim]
    - n_var: Noise standard deviation
    - M_options: List of possible block sizes (e.g., [3, 5, 6, 10])
    - K: Rician factor (default=1 for urban mobile scenario)
    - csi_lag: Number of blocks to lag channel estimate (default=1)
    Returns:
    - Rx_sig_equalized: Equalized received signal
    - batch_snr_db: Average SNR in dB
    """
    batch_size, sequence_length, feature_dim = Tx_sig.shape
    assert feature_dim % 2 == 0, f"feature_dim {feature_dim} must be even for complex symbols"
    complex_length = sequence_length * (feature_dim // 2)

    # Select valid block size M
    valid_M_options = [m for m in M_options if complex_length % m == 0]
    if not valid_M_options:
        raise ValueError(
            f"No M in {M_options} divides complex_length {complex_length}")
    M = random.choice(valid_M_options)
    number_of_blocks = complex_length // M

    # Reshape input signal for block-wise processing
    Tx_sig_reshaped = Tx_sig.view(batch_size, number_of_blocks, M, 2)

    # Generate block-wise fading coefficients
    mean = math.sqrt(K / (K + 1))
    std = math.sqrt(1 / (K + 1))
    H_real_blocks = torch.normal(mean, std,
                                 size=[batch_size, number_of_blocks, 1]).to(
        device)
    H_imag_blocks = torch.normal(mean, std,
                                 size=[batch_size, number_of_blocks, 1]).to(
        device)
    H_blocks = torch.zeros(batch_size, number_of_blocks, 2, 2, device=device)
    H_blocks[:, :, 0, 0] = H_real_blocks[:, :, 0]
    H_blocks[:, :, 0, 1] = -H_imag_blocks[:, :, 0]
    H_blocks[:, :, 1, 0] = H_imag_blocks[:, :, 0]
    H_blocks[:, :, 1, 1] = H_real_blocks[:, :, 0]

    # Apply block-wise fading
    Tx_sig_after_channel_reshaped = torch.matmul(Tx_sig_reshaped, H_blocks)
    Tx_sig_after_channel = Tx_sig_after_channel_reshaped.view(batch_size,
                                                              complex_length, 2)
    Rx_sig, _ = self.AWGN(Tx_sig_after_channel, n_var)  # Ignore AWGN SNR

    # Equalization with lagged channel estimate
    Rx_sig_reshaped = Rx_sig.view(batch_size, number_of_blocks, M, 2)
    H_blocks_lagged = torch.zeros_like(H_blocks)
    # Initial H (e.g., from pilot before transmission)
    H_initial = torch.zeros(batch_size, 2, 2, device=device)
    H_initial[:, 0, 0] = torch.normal(mean, std, size=[batch_size]).to(device)
    H_initial[:, 0, 1] = -torch.normal(mean, std, size=[batch_size]).to(device)
    H_initial[:, 1, 0] = H_initial[:, 0, 1].clone()
    H_initial[:, 1, 1] = H_initial[:, 0, 0].clone()
    # Assign lagged H: block 1 uses H_initial, block 2 uses H_blocks[:,0], etc.
    for i in range(number_of_blocks):
        if i < csi_lag:
            H_blocks_lagged[:, i, :, :] = H_initial
        else:
            H_blocks_lagged[:, i, :, :] = H_blocks[:, i - csi_lag, :, :]
    H_inv_blocks = torch.inverse(H_blocks_lagged)
    Rx_sig_equalized_reshaped = torch.matmul(Rx_sig_reshaped, H_inv_blocks)
    Rx_sig_equalized = Rx_sig_equalized_reshaped.view(batch_size,
                                                      sequence_length,
                                                      feature_dim)

    # Noise power using maximum norm for variability
    noise_power = 2 * n_var ** 2 * torch.max(
        torch.norm(H_inv_blocks, p='fro', dim=[2, 3]) ** 2, dim=1
    )[0]
    snr = 1 / noise_power
    batch_snr = torch.mean(snr).item()
    batch_snr_db = 10 * math.log10(batch_snr)

    return Rx_sig_equalized, batch_snr_db


def estimate_snr_rayleigh(Tx_sig, n_var=0.1, device='cpu', num_runs=100):
    """
    Estimate average SNR and noise power for Rayleigh channel over multiple runs with PowerNormalize.
    - Tx_sig: Input signal tensor (for shape consistency)
    - n_var: Noise variance (default 0.1 as placeholder)
    - device: 'cpu' or 'cuda'
    - num_runs: Number of iterations (default 100)
    """
    batch_size = Tx_sig.shape[0]
    snr_db_list = []
    noise_power_db_list = []

    # Normalize Tx_sig once (mimics training)
    Tx_sig = PowerNormalize(Tx_sig)

    for _ in range(num_runs):
        # Generate H
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)

        # Compute H_inv and noise power
        H_inv = torch.inverse(H)
        noise_power = 2 * n_var ** 2 * torch.norm(H_inv, p='fro') ** 2
        snr = 1 / noise_power  # Signal power assumed 1 after normalization
        snr_db = 10 * math.log10(snr.item())
        noise_power_db = 10 * math.log10(noise_power.item())

        snr_db_list.append(snr_db)
        noise_power_db_list.append(noise_power_db)

    avg_snr_db = np.mean(snr_db_list)
    std_snr_db = np.std(snr_db_list)
    avg_noise_power_db = np.mean(noise_power_db_list)

    print(
        f"Rayleigh - Average SNR (dB): {avg_snr_db:.2f}, Std: {std_snr_db:.2f}")
    print(f"Rayleigh - Average Noise Power (dB): {avg_noise_power_db:.2f}")
    print(
        f"Rayleigh - SNR Range (dB): {min(snr_db_list):.2f} to {max(snr_db_list):.2f}")

    return avg_snr_db, snr_db_list


def estimate_snr_awgn(Tx_sig, n_var=0.1, device='cpu', num_runs=100):
    """
    Estimate average SNR for AWGN channel over multiple runs with PowerNormalize.
    - Tx_sig: Input signal tensor (for shape consistency)
    - n_var: Noise standard deviation (default 0.1)
    - device: 'cpu' or 'cuda'
    - num_runs: Number of iterations (default 100)
    """
    snr_db_list = []
    noise_power_db_list = []

    # Normalize Tx_sig once (mimics training)
    Tx_sig = PowerNormalize(Tx_sig)

    for _ in range(num_runs):
        # AWGN: Noise power = n_var^2, signal power = 1 (after normalization)
        noise_power = n_var ** 2
        snr = 1 / noise_power
        snr_db = 10 * math.log10(snr)
        noise_power_db = 10 * math.log10(noise_power)

        snr_db_list.append(snr_db)
        noise_power_db_list.append(noise_power_db)

    avg_snr_db = np.mean(snr_db_list)
    std_snr_db = np.std(snr_db_list)
    avg_noise_power_db = np.mean(noise_power_db_list)

    print(f"AWGN - Average SNR (dB): {avg_snr_db:.2f}, Std: {std_snr_db:.2f}")
    print(f"AWGN - Average Noise Power (dB): {avg_noise_power_db:.2f}")
    print(
        f"AWGN - SNR Range (dB): {min(snr_db_list):.2f} to {max(snr_db_list):.2f}")

    return avg_snr_db, snr_db_list


def estimate_snr_rician(Tx_sig, n_var=0.1, K=1, device='cpu', num_runs=100):
    """
    Estimate average SNR for Rician channel over multiple runs with PowerNormalize.
    - Tx_sig: Input signal tensor (for shape consistency)
    - n_var: Noise standard deviation (default 0.1)
    - K: Rician factor (default=1)
    - device: 'cpu' or 'cuda'
    - num_runs: Number of iterations (default 100)
    """
    snr_db_list = []
    noise_power_db_list = []

    # Normalize Tx_sig once (mimics training)
    Tx_sig = PowerNormalize(Tx_sig)

    for _ in range(num_runs):
        # Generate Rician channel
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.tensor([[H_real, -H_imag], [H_imag, H_real]], device=device)

        # Compute inverse for equalization
        H_inv = torch.inverse(H)
        noise_power = 2 * n_var ** 2 * torch.norm(H_inv, p='fro') ** 2
        snr = 1 / noise_power  # Signal power = 1 after normalization
        snr_db = 10 * math.log10(snr.item())
        noise_power_db = 10 * math.log10(noise_power.item())

        snr_db_list.append(snr_db)
        noise_power_db_list.append(noise_power_db)

    avg_snr_db = np.mean(snr_db_list)
    std_snr_db = np.std(snr_db_list)
    avg_noise_power_db = np.mean(noise_power_db_list)

    print(f"Rician - Average SNR (dB): {avg_snr_db:.2f}, Std: {std_snr_db:.2f}")
    print(f"Rician - Average Noise Power (dB): {avg_noise_power_db:.2f}")
    print(
        f"Rician - SNR Range (dB): {min(snr_db_list):.2f} to {max(snr_db_list):.2f}")
    # Assuming snr_db_list contains SNR values in dB from your estimation
    negative_snr_count = sum(1 for snr in snr_db_list if snr < 0)
    print(
        f"Number of times SNR < 0 dB: {negative_snr_count} out of {len(snr_db_list)}")
    plt.hist(snr_db_list, bins=100)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frequency')
    plt.show()
    return avg_snr_db, snr_db_list


def estimate_snr_timevaryingrician(Tx_sig, n_var=0.235, M_options=[3, 5, 6, 10],
                                   K=1,
                                   device='cpu', num_runs=9834):
    """
    Estimate average SNR and noise power over multiple runs with PowerNormalize.
    """
    snr_db_list = []
    noise_power_db_list = []

    # Normalize Tx_sig once (mimics training)
    Tx_sig = PowerNormalize(Tx_sig)

    for _ in range(num_runs):
        batch_size, sequence_length, feature_dim = Tx_sig.shape
        complex_length = sequence_length * (feature_dim // 2)
        M = random.choice([m for m in M_options if complex_length % m == 0])
        number_of_blocks = complex_length // M

        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real_blocks = torch.normal(mean, std,
                                     size=[batch_size, number_of_blocks, 1]).to(
            device)
        H_imag_blocks = torch.normal(mean, std,
                                     size=[batch_size, number_of_blocks, 1]).to(
            device)
        H_blocks = torch.zeros(batch_size, number_of_blocks, 2, 2,
                               device=device)
        H_blocks[:, :, 0, 0] = H_real_blocks[:, :, 0]
        H_blocks[:, :, 0, 1] = -H_imag_blocks[:, :, 0]
        H_blocks[:, :, 1, 0] = H_imag_blocks[:, :, 0]
        H_blocks[:, :, 1, 1] = H_real_blocks[:, :, 0]

        H_inv_blocks = torch.inverse(H_blocks)
        noise_power = 2 * n_var ** 2 * torch.mean(
            torch.norm(H_inv_blocks, p='fro', dim=[2, 3]) ** 2, dim=1)
        snr = 1 / noise_power  # Signal power assumed 1 after normalization
        snr_db = 10 * math.log10(torch.mean(snr).item())
        noise_power_db = 10 * math.log10(noise_power.mean().item())

        snr_db_list.append(snr_db)
        noise_power_db_list.append(noise_power_db)

    avg_snr_db = np.mean(snr_db_list)
    std_snr_db = np.std(snr_db_list)
    avg_noise_power_db = np.mean(noise_power_db_list)

    print(
        f"TimeVaryingRician - Average SNR (dB): {avg_snr_db:.2f}, Std: {std_snr_db:.2f}")
    print(
        f"TimeVaryingRician - Average Noise Power (dB): {avg_noise_power_db:.2f}")
    print(
        f"TimeVaryingRician - SNR Range (dB): {min(snr_db_list):.2f} to {max(snr_db_list):.2f}")

    return avg_snr_db, snr_db_list


if __name__ == "__main__":
    # setup_seed(42)
    batch_size, sequence_length, feature_dim = 128, 30, 16
    Tx_sig = torch.randn(batch_size, sequence_length, feature_dim,
                         dtype=torch.float32)

    n_var = 0.316
    M_options = [3, 5, 6, 10]
    device = 'cpu'

    TimeVaryingRician
    avg_snr_db, snr_db_list = estimate_snr_timevaryingrician(
        Tx_sig, n_var=n_var, M_options=M_options, K=1, device=device,
        num_runs=9834
    )

    # Rayleigh
    # avg_snr_db, snr_db_list = estimate_snr_rayleigh(Tx_sig, n_var=n_var,
    #                                                 device=device,
    #                                                 num_runs=9834)

    # AWGN
    # avg_snr_db, snr_db_list = estimate_snr_awgn(Tx_sig, n_var=n_var,
    #                                             device=device,
    #                                             num_runs=9834)

    # Rician
    # avg_snr_db, snr_db_list = estimate_snr_rician(Tx_sig, n_var=n_var,
    #                                               device=device,
    #                                               num_runs=9834)
