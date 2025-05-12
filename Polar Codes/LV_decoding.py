import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
N = 2048                      # Length of the codeword
K = 1024                      # Number of information bits
L = 8                        # List size for LVA
snr_db_range = np.arange(0, 31, 1)  # SNR range in dB
design_snr_dB = 5             # Design SNR
total_bits_per_snr = int(1e6) # Number of bits to simulate per SNR value
ber_threshold = 1e-3          # BER threshold for outage
num_trials = 100              # Number of trials per SNR (reduced for faster simulation)

# Helper Functions
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

# Improved List Viterbi Algorithm (LVA) for Polar Codes
def improved_list_viterbi_decode(r, h, sigma, frozen_bits, G_N, L=16):
    N = len(r)
    n = int(np.log2(N))
    
    # LLR calculation for soft-decision input
    llr = 2 * r * h / (sigma**2)
    
    # Preprocessing: Identify all information bit positions
    info_bit_positions = np.where(~frozen_bits)[0]
    
    # Initialize the list of candidate paths with a single all-zero path
    paths = [{'u': np.zeros(N, dtype=int), 'metric': 0.0}]
    
    # Process each bit position
    for i in range(N):
        new_paths = []
        
        for path in paths:
            u_current = path['u'].copy()
            current_metric = path['metric']
            
            # If frozen bit, we only have one option (bit = 0)
            if frozen_bits[i]:
                u_current[i] = 0
                # Calculate the expected encoded bit
                # We use the property that for polar codes, the i-th bit of encoded x depends on u[0:i+1]
                x_bit = np.mod(np.sum(u_current[:i+1] @ G_N[0:i+1, i]), 2)
                
                # Calculate log-likelihood contribution
                bit_llr = llr[i]
                if x_bit == 1:
                    bit_llr = -bit_llr  # Flip sign if expected bit is 1
                    
                # Update metric (sum of log-likelihoods, more negative is worse)
                # For Viterbi, we use negative log-likelihood as the metric (minimize)
                new_metric = current_metric - bit_llr
                
                new_paths.append({'u': u_current, 'metric': new_metric})
            else:
                # For information bits, try both 0 and 1
                for bit in [0, 1]:
                    u_new = u_current.copy()
                    u_new[i] = bit
                    
                    # Calculate the expected encoded bit
                    x_bit = np.mod(np.sum(u_new[:i+1] @ G_N[0:i+1, i]), 2)
                    
                    # Calculate log-likelihood contribution
                    bit_llr = llr[i]
                    if x_bit == 1:
                        bit_llr = -bit_llr  # Flip sign if expected bit is 1
                        
                    # Update metric
                    new_metric = current_metric - bit_llr
                    
                    new_paths.append({'u': u_new, 'metric': new_metric})
                    
        # Sort paths by metric (lower metric is better in our implementation)
        new_paths.sort(key=lambda p: p['metric'])
        
        # Keep only the L best paths
        paths = new_paths[:L]
    
    # Select the path with the best metric
    best_path_idx = np.argmin([p['metric'] for p in paths])
    u_hat = paths[best_path_idx]['u']
    
    # Re-encode to get the estimated codeword
    x_hat = polar_encode(u_hat, G_N)
    
    return x_hat

def simulate(codeword, snr_dB, frozen_bits, G_N, L):
    snr = 10 ** (snr_dB / 10)
    sigma = np.sqrt(1 / (2 * snr))
    h = np.random.rayleigh(scale=1.0, size=len(codeword))
    bpsk = (1 - 2 * codeword) * h
    noise = sigma * np.random.randn(len(codeword))
    r = bpsk + noise
    
    # Use improved List Viterbi Algorithm
    x_hat = improved_list_viterbi_decode(r, h, sigma, frozen_bits, G_N, L)
    
    return x_hat

def theoretical_rayleigh_ber(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))

# Main Simulation
def main():
    print("Generating polar transform matrix...")
    G_N = generate_polar_transform(int(np.log2(N)))
    G_N = np.mod(G_N, 2).astype(int)
    
    print("Determining frozen bit positions...")
    frozen_bits = get_frozen_bits(N, K, design_snr_dB)

    ber_lva = []
    outage_probability = []
    ber_theory = theoretical_rayleigh_ber(snr_db_range)

    print("Simulating Polar with improved List Viterbi over Rayleigh+AWGN")
    
    for snr_db in tqdm(snr_db_range, desc="SNR"):
        total_errors = 0
        total_bits = 0
        outage_count = 0

        for trial in range(num_trials):
            # Generate random information bits
            info_bits = np.random.randint(0, 2, K)
            
            # Place information bits in non-frozen positions
            u = np.zeros(N, dtype=int)
            u[~frozen_bits] = info_bits
            
            # Encode
            x = polar_encode(u, G_N)
            
            # Transmit and decode
            x_hat = simulate(x, snr_db, frozen_bits, G_N, L)
            
            # Extract decoded information bits
            decoded_info = x_hat[~frozen_bits]
            
            # Count errors
            bit_errors = np.sum(decoded_info != info_bits)
            total_errors += bit_errors
            total_bits += K
            
            # Check for outage
            if bit_errors / K > ber_threshold:
                outage_count += 1
                
            # Print progress for long simulations
            if (trial + 1) % 10 == 0:
                current_ber = total_errors / total_bits
                current_outage = outage_count / (trial + 1)
                print(f"SNR={snr_db} dB | Trial {trial+1}/{num_trials} | Current BER={current_ber:.6e} | Current Outage={current_outage:.4f}")

        # Calculate final BER and outage probability
        ber = total_errors / total_bits
        outage = outage_count / num_trials
        
        # Store results
        ber_lva.append(max(ber, 1e-10))  # Avoid log of zero in plot
        outage_probability.append(outage)

        print(f"SNR={snr_db} dB | Final BER={ber:.6e} | Final Outage={outage:.4f}")

    print("Plotting results...")
    
    # Plotting BER vs SNR with Theoretical Comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ber_lva, 'bo-', label='Polar with List Viterbi')
    plt.semilogy(snr_db_range, ber_theory, 'g--', label='Theoretical BPSK (Rayleigh)')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR: Polar with List Viterbi vs Theoretical Rayleigh BPSK")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('polar_lva_ber.png')
    plt.show()

    # Plotting Outage Probability vs SNR
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, outage_probability, 'ro-', label='Outage Probability')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.title("Outage Probability vs SNR (Threshold BER = 10^-4)")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('polar_lva_outage.png')
    plt.show()

if __name__ == "__main__":
    main()