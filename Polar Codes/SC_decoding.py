import numpy as np
from numpy.random import randn, randint
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm import tqdm  # For the loading bar

# ----- Parameters ----- #
N = 1024               # Length of the polar codeword
K = 512                # Number of information bits
num_bits = int(1e6)    # Total number of bits to simulate
design_snr_dB = 5      # Design SNR in dB for calculating frozen bits
snr_db_range = np.arange(0, 31, 1)  # SNR range from 0 to 30 dB
num_trials = 100        # Number of trials per SNR value
ber_threshold = 1e-2   # BER threshold for outage probability

# ----- Polar Code Construction ----- #
def generate_kernel():
    return np.array([[1, 0], [1, 1]])

def generate_polar_transform(n):
    G = generate_kernel()
    for _ in range(n - 1):
        G = np.kron(G, generate_kernel())
    return G

def calculate_bhattacharyya(N, design_snr_dB):
    snr = 10**(design_snr_dB / 10)
    z = np.zeros(N)
    z[0] = np.exp(-snr)
    for lev in range(int(np.log2(N))):
        B = 2**lev
        for i in range(B):
            T = z[i]
            z[2 * i] = 2 * T - T**2  # Upper channel
            z[2 * i + 1] = T**2      # Lower channel
    return z

def get_frozen_bits(N, K, design_snr_dB):
    z = calculate_bhattacharyya(N, design_snr_dB)
    indices = np.argsort(z)
    info_bits = np.sort(indices[:K])
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_bits] = False
    return frozen_bits

# ----- Polar Encoding ----- #
def polar_encode(u, G_N):
    return np.mod(np.dot(u, G_N), 2)

# ----- Recursive SC Decoder ----- #
def sc_decode_recursive(llr, frozen_bits, u_hat, depth=0):
    N = len(llr)
    if N == 1:
        return np.array([0 if frozen_bits[0] or llr[0] >= 0 else 1])
    
    # LLR for left side
    llr_left = np.sign(llr[:N//2]) * np.sign(llr[N//2:]) * np.minimum(np.abs(llr[:N//2]), np.abs(llr[N//2:]))
    u_left = sc_decode_recursive(llr_left, frozen_bits[:N//2], u_hat, depth + 1)
    
    # LLR for right side
    llr_right = ((1 - 2 * u_left) * llr[:N//2]) + llr[N//2:]
    u_right = sc_decode_recursive(llr_right, frozen_bits[N//2:], u_hat, depth + 1)

    return np.concatenate([u_left ^ u_right, u_right])

def sc_decode(llr, frozen_bits):
    u_hat = np.zeros_like(llr)
    return sc_decode_recursive(llr, frozen_bits, u_hat)

# ----- Channel Simulations ----- #
def simulate_rayleigh_awgn(codeword, snr_dB, frozen_bits, G_N):
    snr = 10**(snr_dB / 10)
    sigma = np.sqrt(1 / (2 * snr))
    
    # Rayleigh fading
    h = np.random.rayleigh(scale=1.0, size=len(codeword))
    bpsk = (1 - 2 * codeword) * h
    
    # AWGN noise
    noise = sigma * randn(len(codeword))
    
    # Received signal y = hX + noise
    r = bpsk + noise
    
    # Compute LLR and decode using SC decoder
    llr = 2 * r * h / (sigma**2)
    u_hat = sc_decode(llr, frozen_bits)
    x_hat = polar_encode(u_hat, G_N)
    
    return x_hat

# ----- Main Simulation ----- #
def main():
    G_N = generate_polar_transform(int(np.log2(N)))
    frozen_bits = get_frozen_bits(N, K, design_snr_dB)
    
    ber_rayleigh_awgn = []
    outage_probability = []
    theoretical_bpsk_rayleigh_awgn = []

    print("Simulating... (can take a few minutes)")
    for snr_db in tqdm(snr_db_range, desc="SNR values", unit="dB"):
        errors_rayleigh_awgn = 0
        total = 0
        outage_count = 0
        
        # 50 trials per SNR value
        for trial in range(num_trials):
            # Random input bits
            u = np.zeros(N, dtype=int)
            info = randint(0, 2, K)
            u[~frozen_bits] = info
            x = polar_encode(u, G_N)
            
            # Simulate through Rayleigh fading and AWGN channel
            x_hat_rayleigh_awgn = simulate_rayleigh_awgn(x, snr_db, frozen_bits, G_N)

            total += K
            errors = np.sum(x_hat_rayleigh_awgn[~frozen_bits] != info)
            errors_rayleigh_awgn += errors
            
            # Check if BER exceeds threshold for outage probability
            if errors / K > ber_threshold:
                outage_count += 1

        # Compute BER for Rayleigh + AWGN channel
        ber_rayleigh_awgn.append(errors_rayleigh_awgn / total)
        
        # Outage Probability
        outage_probability.append(outage_count / num_trials)
        
        # Theoretical BPSK BER for Rayleigh + AWGN
        snr_linear = 10**(snr_db / 10)
        theoretical_bpsk_rayleigh_awgn.append(0.5 * (1 - np.sqrt(snr_linear / (snr_linear + 1))))

        print(f"SNR={snr_db} dB | BER_Rayleigh_AWGN={ber_rayleigh_awgn[-1]:.4e} | Outage Probability={outage_probability[-1]:.4f}")
        ber_rayleigh_awgn = [max(ber, 1e-10) for ber in ber_rayleigh_awgn]

    # Plotting BER vs. SNR
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ber_rayleigh_awgn, 'bo-', label='Polar SC (Rayleigh + AWGN)')
    plt.semilogy(snr_db_range, theoretical_bpsk_rayleigh_awgn, 'g--', label='BPSK Rayleigh + AWGN (Theoretical)')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER Performance of Polar Code (SC Decoder) with Rayleigh Fading + AWGN")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting Outage Probability vs. SNR
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db_range, outage_probability, 'ro-', label='Outage Probability')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.title("Outage Probability vs. SNR for Polar Code (SC Decoder) with Rayleigh Fading + AWGN")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
