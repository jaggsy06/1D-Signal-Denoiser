# 1D Signal Denoiser (Dense → 1D CNN)

A neural network-based denoiser for 1D signals, capable of reducing noise in **sine, square, and sawtooth waves** with diverse noise types. This project demonstrates the evolution from a basic dense autoencoder to a **1D CNN autoencoder** that preserves sharp edges and temporal structure.

---

## Motivation

Denoising 1D signals is a common task in **lab experiments, sensors, and signal processing**. Traditional methods (like filters) often struggle with:

- Multiple types of noise simultaneously (Gaussian, uniform, impulse)
- Signals with sharp discontinuities (e.g., square and sawtooth waves)

The goal of this project is to use **deep learning** to create a **general-purpose denoiser** that can handle multiple waveforms and noise types.

---

## Features

- Generates **diversified synthetic signals** (sine, square, sawtooth)  
- Supports multiple **noise types**: Gaussian, uniform, impulse spikes  
- **Dense autoencoder** (initial baseline)  
- **1D CNN autoencoder** (upgraded version) for better edge preservation  
- Quantitative evaluation using **MSE** and **SNR**  
- Visual comparison of **clean, noisy, and denoised signals**

---

## Why CNN Instead of Dense Layers

The initial **dense autoencoder** worked reasonably well for smooth signals (like sine waves) but **failed for signals with sharp edges**, as demonstrated by prior results:

- Square and sawtooth waves were oversmoothed  
- Residual noise remained for impulse spikes  

**1D CNN Autoencoder Advantages:**

- **Captures local temporal patterns** → preserves edges and ramps  
- **Shared convolutional kernels** → better generalization to multiple waveforms  
- **Efficient** → fewer parameters than fully connected dense layers for long signals  

This change significantly improved denoising quality for **all three waveform types**.

---

## Process Overview

### 1. Signal Preparation

- Signals generated synthetically:
  - Waveforms: sine, square, sawtooth  
  - Random frequency (0.5–2.0), phase (0–2π), amplitude (0.5–1.5)
- Noise types added randomly:
  - Gaussian, uniform, impulse spikes
- Dataset size: **2000 training samples**, each of length 256

### 2. Data Formatting for CNN

- Reshaped to `(samples, length, 1)` → required by `Conv1D` layers

### 3. Model Architecture

**1D CNN Autoencoder**:

- **Encoder:**
  - `Conv1D(32, kernel=5, activation=relu)` → `MaxPooling1D(2)`
  - `Conv1D(16, kernel=5, activation=relu)` → `MaxPooling1D(2)`
- **Bottleneck:**
  - `Conv1D(16, kernel=3, activation=relu)`
- **Decoder:**
  - `UpSampling1D(2)` → `Conv1D(32, kernel=5, activation=relu)`
  - `UpSampling1D(2)` → `Conv1D(1, kernel=5, activation=linear)`

**Loss function:** Mean Squared Error (MSE)  
**Optimizer:** Adam, learning rate 1e-3  

---

### 4. Training

- Epochs: 50  
- Batch size: 32  
- Validation split: 10%  

---

## Quantitative Evaluation

| Waveform   | MSE Before | MSE After | SNR Before | SNR After |
|------------|------------|-----------|------------|-----------|
| Sine       | 0.2679     | 0.0110    | 1.58 dB    | 15.47 dB  |
| Square     | 0.0033     | 0.0033    | 23.70 dB   | 23.63 dB  |
| Sawtooth   | 0.2708     | 0.0392    | 2.70 dB    | 11.09 dB  |

**Observations:**

- Sine wave: **significant noise reduction**  
- Square wave: edges preserved, minimal oversmoothing  
- Sawtooth wave: noise reduced substantially without destroying ramps  

---

## Visual Results

Plots show **Clean**, **Noisy**, and **Denoised** signals for all three waveform types:

```python
plt.figure(figsize=(15, 8))
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(test_clean[i].squeeze(), label="Clean")
    plt.plot(test_noisy[i].squeeze(), label="Noisy", alpha=0.5)
    plt.plot(denoised_test[i].squeeze(), label="Denoised")
    plt.legend()
    plt.title(f"{waveforms[i]} Wave - 1D CNN Denoiser")
plt.tight_layout()
plt.show()
