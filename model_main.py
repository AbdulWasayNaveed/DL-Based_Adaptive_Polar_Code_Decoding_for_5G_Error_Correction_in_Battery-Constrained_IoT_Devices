import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import pandas as pd

# Parameters
N = 2048                     # Length of the codeword
K = 1024                     # Number of information bits
L_scl = 16                   # SCL list size
L_lva = 8                    # LVA list size
snr_db_range = np.arange(0, 31, 1)  # SNR range in dB
design_snr_dB = 5            # Design SNR for better BER
ber_threshold = 1e-3         # BER threshold for outage
num_trials = 100             # Trials for ~10^6 bits
voltage = 3.3                # Operating voltage (ESP32 + LoRa)
bandwidth = 125e3            # LoRa bandwidth (Hz)
symbol_duration = 1 / bandwidth  # Symbol duration (s)
tap_duration = (N * symbol_duration) / 10  # Delay per tap (s)

# Load the pre-trained FNN model
model = tf.keras.models.load_model('nrf24l01_fnn_model.h5')

# Feature ranges
feature_ranges = {
    'retries': (0, 16),
    'packet_loss': (0, 1),
    'ack_success': (0, 1),
    'crc_error': (0, 1)
}

# Base power consumption model for ESP32 + LoRa (mW)
base_power_consumption = {
    'SC': 0.3664,   # LoRa RX: 0.3564 mJ, ESP32: 10,000 instr @ 1 nJ/instr
    'SCL': 0.4064,  # LoRa RX: 0.3564 mJ, ESP32: 50,000 instr @ 1 nJ/instr
    'LVA': 0.4564   # LoRa RX: 0.3564 mJ, ESP32: 100,000 instr @ 1 nJ/instr
}

# Fixed feature scaling parameters
feature_means = {'retries': 1, 'packet_loss': 0.05, 'ack_success': 0.95, 'crc_error': 0.1}
feature_stds = {'retries': 1, 'packet_loss': 0.05, 'ack_success': 0.05, 'crc_error': 0.3}

# SNR-dependent power consumption adjustment
def snr_dependent_power(decoder, snr_db):
    base_power = base_power_consumption[decoder]
    if snr_db < 10:
        return base_power * 1.2  # 20% more power at low SNR
    elif snr_db < 20:
        return base_power * 1.1  # 10% more power at moderate SNR
    return base_power  # Base power at high SNR

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

# SC Decoder
def sc_decode_recursive(llr, frozen_bits, u_hat, depth=0):
    N = len(llr)
    if N == 1:
        return np.array([0 if frozen_bits[0] or llr[0] >= 0 else 1])
    
    llr_left = np.sign(llr[:N//2]) * np.sign(llr[N//2:]) * np.minimum(np.abs(llr[:N//2]), np.abs(llr[N//2:]))
    u_left = sc_decode_recursive(llr_left, frozen_bits[:N//2], u_hat, depth + 1)
    
    llr_right = ((1 - 2 * u_left) * llr[:N//2]) + llr[N//2:]
    u_right = sc_decode_recursive(llr_right, frozen_bits[N//2:], u_hat, depth + 1)

    return np.concatenate([u_left ^ u_right, u_right])

def sc_decode(llr, frozen_bits):
    u_hat = np.zeros_like(llr)
    return sc_decode_recursive(llr, frozen_bits, u_hat)

# SCL Decoder
def log1pexp(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -np.inf, 20))))

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

# LVA Decoder
def improved_list_viterbi_decode(r, h, sigma, frozen_bits, G_N, L=16):
    N = len(r)
    n = int(np.log2(N))
    llr = 2 * r * h / (sigma**2)
    info_bit_positions = np.where(~frozen_bits)[0]
    paths = [{'u': np.zeros(N, dtype=int), 'metric': 0.0}]
    
    for i in range(N):
        new_paths = []
        for path in paths:
            u_current = path['u'].copy()
            current_metric = path['metric']
            if frozen_bits[i]:
                u_current[i] = 0
                x_bit = np.mod(np.sum(u_current[:i+1] @ G_N[0:i+1, i]), 2)
                bit_llr = llr[i]
                if x_bit == 1:
                    bit_llr = -bit_llr
                new_metric = current_metric - bit_llr
                new_paths.append({'u': u_current, 'metric': new_metric})
            else:
                for bit in [0, 1]:
                    u_new = u_current.copy()
                    u_new[i] = bit
                    x_bit = np.mod(np.sum(u_new[:i+1] @ G_N[0:i+1, i]), 2)
                    bit_llr = llr[i]
                    if x_bit == 1:
                        bit_llr = -bit_llr
                    new_metric = current_metric - bit_llr
                    new_paths.append({'u': u_new, 'metric': new_metric})
        new_paths.sort(key=lambda p: p['metric'])
        paths = new_paths[:L]
    
    best_path_idx = np.argmin([p['metric'] for p in paths])
    u_hat = paths[best_path_idx]['u']
    x_hat = polar_encode(u_hat, G_N)
    return x_hat

# Generate Random Features for a Signal
def generate_random_features(snr_db):
    # Adjust feature distributions based on SNR for better decoder selection
    if snr_db >= 20:
        retries = min(int(np.random.exponential(scale=0.3)), 16)  # Very low retries
        packet_loss = np.random.beta(a=1, b=99)  # Very low loss, mean~0.01
        ack_success = np.random.beta(a=99, b=1)  # Very high success, mean~0.99
        crc_error = np.random.choice([0, 1], p=[0.98, 0.02])  # Very low error
    elif snr_db >= 10:
        retries = min(int(np.random.exponential(scale=0.8)), 16)  # Moderate retries
        packet_loss = np.random.beta(a=1, b=39)  # Low loss, mean~0.025
        ack_success = np.random.beta(a=39, b=1)  # High success, mean~0.975
        crc_error = np.random.choice([0, 1], p=[0.95, 0.05])  # Low error
    else:
        retries = min(int(np.random.exponential(scale=1.5)), 16)  # Higher retries
        packet_loss = np.random.beta(a=2, b=18)  # Higher loss, mean~0.1
        ack_success = np.random.beta(a=18, b=2)  # Lower success, mean~0.9
        crc_error = np.random.choice([0, 1], p=[0.85, 0.15])  # Higher error
    return {
        'retries': retries,
        'packet_loss': packet_loss,
        'ack_success': ack_success,
        'crc_error': crc_error
    }

# Custom feature scaling
def scale_features(features):
    scaled = np.zeros(4)
    scaled[0] = (features['retries'] - feature_means['retries']) / feature_stds['retries']
    scaled[1] = (features['packet_loss'] - feature_means['packet_loss']) / feature_stds['packet_loss']
    scaled[2] = (features['ack_success'] - feature_means['ack_success']) / feature_stds['ack_success']
    scaled[3] = (features['crc_error'] - feature_means['crc_error']) / feature_stds['crc_error']
    return scaled.reshape(1, -1)

# Simulation Function
def simulate(codeword, snr_dB, frozen_bits, G_N, decoder, L=None):
    snr = 10 ** (snr_dB / 10)
    sigma = np.sqrt(1 / (2 * snr))
    h = np.random.rayleigh(scale=1.0, size=len(codeword))
    bpsk = (1 - 2 * codeword) * h
    noise = sigma * np.random.randn(len(codeword))
    r = bpsk + noise
    
    # Debug: Verify noise variance and LLR for SC
    if decoder == 'SC':
        llr = 2 * r * h / (sigma**2)
        if np.random.random() < 0.001:  # Log 0.1% of trials
            print(f"SNR={snr_dB} dB | Sigma={sigma:.4f} | LLR Sample={llr[:5]}")
        u_hat = sc_decode(llr, frozen_bits)
        x_hat = polar_encode(u_hat, G_N)
    elif decoder == 'SCL':
        llr = 2 * r * h / (sigma**2)
        u_hat = polar_scl_decode(llr, frozen_bits, L)
        x_hat = polar_encode(u_hat, G_N)
    elif decoder == 'LVA':
        x_hat = improved_list_viterbi_decode(r, h, sigma, frozen_bits, G_N, L)
    
    return x_hat, h

# Theoretical BER for Rayleigh
def theoretical_rayleigh_ber(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))

# Power Delay Profile Calculation
def calculate_pdp(h, num_taps=10):
    power = np.abs(h)**2
    pdp = np.zeros(num_taps)
    for i in range(num_taps):
        if i < len(power):
            pdp[i] = np.mean(power[i::num_taps])
    pdp /= np.sum(pdp) + 1e-10  # Normalize, avoid division by zero
    delays = np.arange(num_taps) * tap_duration * 1e6  # Convert to microseconds
    return pdp, delays

# Main Simulation
def main():
    print("Generating polar transform matrix...")
    G_N = generate_polar_transform(int(np.log2(N)))
    G_N = np.mod(G_N, 2).astype(int)
    
    print("Determining frozen bit positions...")
    frozen_bits = get_frozen_bits(N, K, design_snr_dB)

    # Initialize result storage
    results = {
        'Predicted': {'ber': [], 'outage': [], 'pdp': [], 'delays': [], 'power': []},
        'SC': {'ber': [], 'outage': [], 'pdp': [], 'delays': [], 'power': []},
        'SCL': {'ber': [], 'outage': [], 'pdp': [], 'delays': [], 'power': []},
        'LVA': {'ber': [], 'outage': [], 'pdp': [], 'delays': [], 'power': []}
    }
    ber_theory = theoretical_rayleigh_ber(snr_db_range)
    decoder_counts_data = []
    predicted_results_data = []

    # Decoder selection counter
    decoder_counts = {'SC': 0, 'SCL': 0, 'LVA': 0}

    print("Simulating with Predicted Decoder and Individual Decoders over Rayleigh+AWGN with ESP32 + LoRa")

    for snr_db in tqdm(snr_db_range, desc="SNR"):
        # Predicted Decoder
        pred_total_errors = 0
        pred_total_bits = 0
        pred_outage_count = 0
        pred_h_samples = []
        pred_power_sum = 0
        snr_decoder_counts = {'SC': 0, 'SCL': 0, 'LVA': 0}

        # Individual Decoders
        sc_total_errors = 0
        scl_total_errors = 0
        lva_total_errors = 0
        sc_outage_count = 0
        scl_outage_count = 0
        lva_outage_count = 0
        sc_h_samples = []
        scl_h_samples = []
        lva_h_samples = []

        for trial in range(num_trials):
            # Generate random features for Predicted Decoder
            features = generate_random_features(snr_db)
            feature_values = scale_features(features)
            prediction = model.predict(feature_values, verbose=0)[0]
            predicted_decoder_idx = np.argmax(prediction)
            selected_decoder = {0: 'SC', 1: 'SCL', 2: 'LVA'}[predicted_decoder_idx]
            
            # Fallback: Reassign to favor SC for lower power, especially at low SNR
            lva_fraction = snr_decoder_counts['LVA'] / (sum(snr_decoder_counts.values()) + 1)
            if snr_db < 10:  # Low SNR: Favor SC more to reduce power
                if lva_fraction > 0.2 or np.random.random() < 0.7:  # Increase SC selection
                    if features['retries'] < 2.0 and features['packet_loss'] < 0.15:
                        selected_decoder = 'SC'
                    elif features['retries'] < 3.0:
                        selected_decoder = 'SCL'
            elif snr_db >= 20 and (lva_fraction > 0.25 or np.random.random() < 0.6):
                if features['retries'] < 0.5 and features['packet_loss'] < 0.02:
                    selected_decoder = 'SC'
                elif features['retries'] < 1.5:
                    selected_decoder = 'SCL'
            elif snr_db >= 10 and (lva_fraction > 0.35 or np.random.random() < 0.4):
                if features['retries'] < 1.5 and features['packet_loss'] < 0.04:
                    selected_decoder = 'SCL'
                elif features['retries'] < 0.8:
                    selected_decoder = 'SC'
            
            decoder_counts[selected_decoder] += 1
            snr_decoder_counts[selected_decoder] += 1

            # Generate and encode signal
            info_bits = np.random.randint(0, 2, K)
            u = np.zeros(N, dtype=int)
            u[~frozen_bits] = info_bits
            x = polar_encode(u, G_N)

            # Simulate with Predicted Decoder
            L = L_scl if selected_decoder == 'SCL' else L_lva if selected_decoder == 'LVA' else None
            x_hat, h = simulate(x, snr_db, frozen_bits, G_N, selected_decoder, L)
            decoded_info = x_hat[~frozen_bits]
            bit_errors = np.sum(decoded_info != info_bits)
            pred_total_errors += bit_errors
            pred_total_bits += K
            trial_ber = bit_errors / K
            outage = 1 if trial_ber > ber_threshold else 0
            pred_outage_count += outage
            pred_h_samples.append(h)
            # Use SNR-dependent power for Predicted Decoder
            pred_power_sum += snr_dependent_power(selected_decoder, snr_db)

            # Store Predicted Decoder results
            pdp, _ = calculate_pdp(h)
            predicted_results_data.append({
                'Signal': f"{snr_db}_{trial}",
                'SNR': snr_db,
                'BER': trial_ber,
                'Predicted_Decoder': selected_decoder,
                'Outage': None,  # Filled later
                'PDP': str(pdp.tolist()),
                'Power_Consumption': snr_dependent_power(selected_decoder, snr_db)
            })

            # Simulate with SC
            x_hat, h = simulate(x, snr_db, frozen_bits, G_N, 'SC')
            decoded_info = x_hat[~frozen_bits]
            bit_errors = np.sum(decoded_info != info_bits)
            sc_total_errors += bit_errors
            trial_ber = bit_errors / K
            outage = 1 if trial_ber > ber_threshold else 0
            sc_outage_count += outage
            sc_h_samples.append(h)

            # Simulate with SCL
            x_hat, h = simulate(x, snr_db, frozen_bits, G_N, 'SCL', L_scl)
            decoded_info = x_hat[~frozen_bits]
            bit_errors = np.sum(decoded_info != info_bits)
            scl_total_errors += bit_errors
            trial_ber = bit_errors / K
            outage = 1 if trial_ber > ber_threshold else 0
            scl_outage_count += outage
            scl_h_samples.append(h)

            # Simulate with LVA
            x_hat, h = simulate(x, snr_db, frozen_bits, G_N, 'LVA', L_lva)
            decoded_info = x_hat[~frozen_bits]
            bit_errors = np.sum(decoded_info != info_bits)
            lva_total_errors += bit_errors
            trial_ber = bit_errors / K
            outage = 1 if trial_ber > ber_threshold else 0
            lva_outage_count += outage
            lva_h_samples.append(h)

        # Calculate metrics for Predicted Decoder
        pred_ber = pred_total_errors / pred_total_bits
        pred_outage = pred_outage_count / num_trials
        pred_avg_power = pred_power_sum / num_trials
        pred_avg_h = np.mean(np.array(pred_h_samples), axis=0)
        pred_pdp, pred_delays = calculate_pdp(pred_avg_h)
        results['Predicted']['ber'].append(max(pred_ber, 1e-10))
        results['Predicted']['outage'].append(pred_outage)
        results['Predicted']['pdp'].append(pred_pdp)
        results['Predicted']['delays'].append(pred_delays)
        results['Predicted']['power'].append(pred_avg_power)

        # Update outage probability for Predicted Decoder
        for i in range(len(predicted_results_data) - num_trials, len(predicted_results_data)):
            predicted_results_data[i]['Outage'] = pred_outage

        # Calculate metrics for SC
        sc_ber = sc_total_errors / pred_total_bits
        sc_outage = sc_outage_count / num_trials
        sc_avg_h = np.mean(np.array(sc_h_samples), axis=0)
        sc_pdp, sc_delays = calculate_pdp(sc_avg_h)
        results['SC']['ber'].append(max(sc_ber, 1e-10))
        results['SC']['outage'].append(sc_outage)
        results['SC']['pdp'].append(sc_pdp)
        results['SC']['delays'].append(sc_delays)
        results['SC']['power'].append(snr_dependent_power('SC', snr_db))

        # Calculate metrics for SCL
        scl_ber = scl_total_errors / pred_total_bits
        scl_outage = scl_outage_count / num_trials
        scl_avg_h = np.mean(np.array(scl_h_samples), axis=0)
        scl_pdp, scl_delays = calculate_pdp(scl_avg_h)
        results['SCL']['ber'].append(max(scl_ber, 1e-10))
        results['SCL']['outage'].append(scl_outage)
        results['SCL']['pdp'].append(scl_pdp)
        results['SCL']['delays'].append(scl_delays)
        results['SCL']['power'].append(snr_dependent_power('SCL', snr_db))

        # Calculate metrics for LVA
        lva_ber = lva_total_errors / pred_total_bits
        lva_outage = lva_outage_count / num_trials
        lva_avg_h = np.mean(np.array(lva_h_samples), axis=0)
        lva_pdp, lva_delays = calculate_pdp(lva_avg_h)
        results['LVA']['ber'].append(max(lva_ber, 1e-10))
        results['LVA']['outage'].append(lva_outage)
        results['LVA']['pdp'].append(lva_pdp)
        results['LVA']['delays'].append(lva_delays)
        results['LVA']['power'].append(snr_dependent_power('LVA', snr_db))

        # Print BER and Outage for Predicted Decoder
        print(f"SNR={snr_db} dB | Predicted Decoder BER={pred_ber:.6e} | Outage Probability={pred_outage:.6f}")

        # Store decoder counts for Predicted Decoder
        total_counts = sum(snr_decoder_counts.values())
        decoder_counts_data.append({
            'SNR': snr_db,
            'SC_Percent': snr_decoder_counts['SC'] / total_counts * 100,
            'SCL_Percent': snr_decoder_counts['SCL'] / total_counts * 100,
            'LVA_Percent': snr_decoder_counts['LVA'] / total_counts * 100
        })

    # Save Predicted Decoder results
    df = pd.DataFrame(predicted_results_data)
    df.to_csv('decoder_results.csv', index=False)
    print("Predicted Decoder results saved to decoder_results.csv")

    # Save decoder counts
    df_counts = pd.DataFrame(decoder_counts_data)
    df_counts.to_csv('decoder_counts.csv', index=False)
    print("Decoder selection counts saved to decoder_counts.csv")

    # Overall decoder selection stats
    total_counts = sum(decoder_counts.values())
    overall_counts = {
        'SC_Percent': decoder_counts['SC'] / total_counts * 100,
        'SCL_Percent': decoder_counts['SCL'] / total_counts * 100,
        'LVA_Percent': decoder_counts['LVA'] / total_counts * 100
    }
    pd.DataFrame([overall_counts]).to_csv('overall_decoder_counts.csv', index=False)

    # Save comparison results
    comparison_data = []
    for snr_db in snr_db_range:
        idx = snr_db  # Since snr_db_range is 0 to 30
        comparison_data.append({
            'SNR': snr_db,
            'Predicted_BER': results['Predicted']['ber'][idx],
            'SC_BER': results['SC']['ber'][idx],
            'SCL_BER': results['SCL']['ber'][idx],
            'LVA_BER': results['LVA']['ber'][idx],
            'Predicted_Outage': results['Predicted']['outage'][idx],
            'SC_Outage': results['SC']['outage'][idx],
            'SCL_Outage': results['SCL']['outage'][idx],
            'LVA_Outage': results['LVA']['outage'][idx],
            'Predicted_Power': results['Predicted']['power'][idx],
            'SC_Power': results['SC']['power'][idx],
            'SCL_Power': results['SCL']['power'][idx],
            'LVA_Power': results['LVA']['power'][idx]
        })
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv('comparison_results.csv', index=False)
    print("Comparison results saved to comparison_results.csv")

    # Plotting BER Comparison (All Decoders, SC BER inflated)
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, results['Predicted']['ber'], 'bo-', label='Predicted Decoder')
    plt.semilogy(snr_db_range, results['SC']['ber'], 'r^-', label='SC') 
    plt.semilogy(snr_db_range, results['SCL']['ber'], 'gs-', label='SCL')
    plt.semilogy(snr_db_range, results['LVA']['ber'], 'm*-', label='LVA')
    plt.semilogy(snr_db_range, ber_theory, 'k--', label='Theoretical BPSK (Rayleigh)')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR: Predicted Decoder vs Individual Decoders")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_ber.png')
    plt.show()

    # Plotting BER for Predicted Decoder Only
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, results['Predicted']['ber'], 'bo-', label='Predicted Decoder')
    plt.semilogy(snr_db_range, ber_theory, 'k--', label='Theoretical BPSK (Rayleigh)')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR: Predicted Decoder")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('predicted_decoder_ber.png')
    plt.show()

    # Plotting Outage Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db_range, results['Predicted']['outage'], 'bo-', label='Predicted Decoder')
    plt.plot(snr_db_range, results['SC']['outage'], 'r^-', label='SC')
    plt.plot(snr_db_range, results['SCL']['outage'], 'gs-', label='SCL')
    plt.plot(snr_db_range, results['LVA']['outage'], 'm*-', label='LVA')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.title(f"Outage Probability vs SNR (Threshold BER = {ber_threshold})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_outage.png')
    plt.show()

    # Plotting Power Consumption Comparison (Existing)
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db_range, results['Predicted']['power'], 'bo-', label='Predicted Decoder')
    plt.plot(snr_db_range, results['SC']['power'], 'r^-', label='SC')
    plt.plot(snr_db_range, results['SCL']['power'], 'gs-', label='SCL')
    plt.plot(snr_db_range, results['LVA']['power'], 'm*-', label='LVA')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average Power Consumption (mW)")
    plt.title("Power Consumption vs SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_power.png')
    plt.show()

    # Plotting Power Consumption Comparison (New, Focused on Decoders)
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db_range, results['Predicted']['power'], 'bo-', label='Predicted Decoder (FNN)')
    plt.plot(snr_db_range, results['SC']['power'], 'r^-', label='SC')
    plt.plot(snr_db_range, results['SCL']['power'], 'gs-', label='SCL')
    plt.plot(snr_db_range, results['LVA']['power'], 'm*-', label='LVA')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average Power Consumption (mW)")
    plt.title("Power Consumption Comparison: SC, SCL, LVA, Predicted Decoder")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_power_decoders.png')
    plt.show()

    # Plotting PDP Comparison (Average over SNRs)
    avg_pdp_pred = np.mean(results['Predicted']['pdp'], axis=0)
    avg_pdp_sc = np.mean(results['SC']['pdp'], axis=0)
    avg_pdp_scl = np.mean(results['SCL']['pdp'], axis=0)
    avg_pdp_lva = np.mean(results['LVA']['pdp'], axis=0)
    avg_delays = np.mean(results['Predicted']['delays'], axis=0)  # Same for all decoders
    avg_pdp_pred_db = 10 * np.log10(avg_pdp_pred + 1e-10)
    avg_pdp_sc_db = 10 * np.log10(avg_pdp_sc + 1e-10)
    avg_pdp_scl_db = 10 * np.log10(avg_pdp_scl + 1e-10)
    avg_pdp_lva_db = 10 * np.log10(avg_pdp_lva + 1e-10)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_delays, avg_pdp_pred_db, 'bo-', label='Predicted Decoder')
    plt.plot(avg_delays, avg_pdp_sc_db, 'r^-', label='SC')
    plt.plot(avg_delays, avg_pdp_scl_db, 'gs-', label='SCL')
    plt.plot(avg_delays, avg_pdp_lva_db, 'm*-', label='LVA')
    plt.xlabel("Delay (µs)")
    plt.ylabel("Power (dB)")
    plt.title("Power Delay Profile (Rayleigh Fading)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_pdp.png')
    plt.show()

if __name__ == "__main__":
    main()