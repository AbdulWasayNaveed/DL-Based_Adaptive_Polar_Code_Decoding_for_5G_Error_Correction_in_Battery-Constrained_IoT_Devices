# FNN-Based Adaptive Polar Code Decoding for 5G in Battery-Constrained IoT Devices

## Overview

This project implements an intelligent error correction system using **Polar Codes** for 5G communication systems in **battery-constrained embedded devices**. It integrates **machine learning (ML)**, specifically a **Feedforward Neural Network (FNN)**, to adaptively select the optimal decoding algorithm in real-time based on channel conditions, aiming to balance **error performance** and **power efficiency**.

The system is simulated in Python using a realistic wireless channel model (Rayleigh fading + AWGN), and supports deployment on embedded hardware like **ESP32 + LoRa SX1278** for low-power, long-range IoT applications.

## Key Features

- Adaptive decoder selection using a trained FNN
- Polar encoding and decoding (SC, SCL, LVA)
- Channel modeling with BPSK modulation, Rayleigh fading, and AWGN
- BER, outage probability, power consumption, and PDP evaluation
- Simulation across a 0–30 dB SNR range
- Power-aware decoding with energy models for embedded systems

## Project Structure

- `model_main.py` – Main simulation script, decoder implementations, FNN inference, performance evaluation
- `nrf24l01_fnn_model.h5` – Trained Feedforward Neural Network model (not included; must be provided)
- `Theseis_DL Based 5G Error Correction using Polar codes.docx` – Full thesis/documentation (see `/docs`)

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- tqdm

Install dependencies:

```bash
pip install tensorflow numpy pandas matplotlib tqdm
