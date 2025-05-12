import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
N = 2048                     # Length of the codeword
K = 1024                     # Number of information bits
L = 16                        # SCL list size
snr_db_range = np.arange(0, 31, 1)  # SNR range in dB
design_snr_dB = 5            # Design SNR
total_bits_per_snr = int(1e6) # Number of bits to simulate per SNR value
ber_threshold = 1e-3         # BER threshold for outage
num_trials = 150             # Number of trials per SNR

# Helper Functions
def log1pexp(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def generate_kernel():
    return np.array([[1, 0], [1, 1]])

def generate_polar_transform(n):
    G = generate_kernel()
    for _ in range(n - 1):
        G = np.kron(G, generate_kernel())
    return G

def calculate_bhattacharyya(N, design_snr_dB):
    snr = 10 ** (design_snr_dB / 10)
    z = np.zeros(N)
    z[0] = np.exp(-snr)
    for lev in range(int(np.log2(N))):
        B = 2**lev
        for i in range(B):
            T = z[i]
            z[2*i] = 2*T - T**2
            z[2*i+1] = T**2
    return z

def get_frozen_bits(N, K, design_snr_dB):
    z = calculate_bhattacharyya(N, design_snr_dB)
    indices = np.argsort(z)
    info_bits = np.sort(indices[:K])
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_bits] = False
    return frozen_bits

def polar_encode(u, G_N):
    return np.mod(np.dot(u, G_N), 2)

def polar_scl_decode(llr, frozen_bits, L=4):
    N = len(llr)
    paths = [{'u': np.zeros(N, dtype=int), 'pm': 0.0}]
    for i in range(N):
        new_paths = []
        for path in paths:
            u, pm = path['u'], path['pm']
            llr_val = llr[i]
            if frozen_bits[i]:
                bit = 0
                pm_new = pm + log1pexp(np.abs(llr_val))
                u_new = np.copy(u)
                u_new[i] = bit
                new_paths.append({'u': u_new, 'pm': pm_new})
            else:
                for bit in [0, 1]:
                    if bit == 0:
                        pm_new = pm + log1pexp(llr_val)
                    else:
                        pm_new = pm + llr_val + log1pexp(llr_val)
                    u_new = np.copy(u)
                    u_new[i] = bit
                    new_paths.append({'u': u_new, 'pm': pm_new})
        new_paths.sort(key=lambda p: p['pm'])
        paths = new_paths[:L]
    best_path = min(paths, key=lambda p: p['pm'])
    return best_path['u']

def simulate(codeword, snr_dB, frozen_bits, G_N, L):
    snr = 10 ** (snr_dB / 10)
    sigma = np.sqrt(1 / (2 * snr))
    h = np.random.rayleigh(scale=1.0, size=len(codeword))
    bpsk = (1 - 2 * codeword) * h
    noise = sigma * np.random.randn(len(codeword))
    r = bpsk + noise
    llr = 2 * r * h / (sigma**2)
    u_hat = polar_scl_decode(llr, frozen_bits, L)
    x_hat = polar_encode(u_hat, G_N)
    return x_hat

def theoretical_rayleigh_ber(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))

# Main Simulation
def main():
    G_N = generate_polar_transform(int(np.log2(N)))
    G_N = np.mod(G_N, 2).astype(int)
    frozen_bits = get_frozen_bits(N, K, design_snr_dB)

    ber_scl = []
    outage_probability = []
    ber_theory = theoretical_rayleigh_ber(snr_db_range)

    print("Simulating Polar SCL over Rayleigh+AWGN")
    
    for snr_db in tqdm(snr_db_range, desc="SNR"):
        total_errors = 0
        total_blocks = 0
        outage_count = 0

        for _ in range(num_trials):
            u = np.zeros(N, dtype=int)
            info_bits = np.random.randint(0, 2, K)
            u[~frozen_bits] = info_bits
            x = polar_encode(u, G_N)
            x_hat = simulate(x, snr_db, frozen_bits, G_N, L)
            decoded_info = x_hat[~frozen_bits]
            errors = np.sum(decoded_info != info_bits)
            total_errors += errors
            total_blocks += 1
            if errors / K > ber_threshold:
                outage_count += 1

        ber = total_errors / (total_blocks * K)
        outage = outage_count / num_trials
        ber_scl.append(max(ber, 1e-10))
        outage_probability.append(outage)

        print(f"SNR={snr_db} dB | BER={ber:.4e} | Outage={outage:.4f}")

    # Plotting BER vs SNR with Theoretical Comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ber_scl, 'bo-', label='Simulated Polar SCL')
    plt.semilogy(snr_db_range, ber_theory, 'g--', label='Theoretical BPSK (Rayleigh)')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR: Polar SCL vs Theoretical Rayleigh BPSK")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting Outage Probability vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db_range, outage_probability, 'ro-', label='Outage Probability')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.title("Outage Probability vs SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
