import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# 1. Generate varied synthetic signals

def generate_signals(n_samples=2000, signal_length=256):
    x = np.linspace(0, 4 * np.pi, signal_length)
    clean_signals = []
    noisy_signals = []

    for _ in range(n_samples):
        # Randomly choose signal type
        signal_type = np.random.choice(["sine", "square", "sawtooth"])
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.5, 1.5)

        if signal_type == "sine":
            clean = amplitude * np.sin(freq * x + phase)
        elif signal_type == "square":
            clean = amplitude * np.sign(np.sin(freq * x + phase))
        else:  # sawtooth
            clean = amplitude * (2 * ((freq*x/(2*np.pi) + phase/(2*np.pi)) % 1) - 1)

        # Random noise
        noise_type = np.random.choice(["gaussian", "uniform", "impulse"])
        if noise_type == "gaussian":
            noise = np.random.normal(0, 0.5, signal_length)
        elif noise_type == "uniform":
            noise = np.random.uniform(-0.5, 0.5, signal_length)
        else:  # impulse noise
            noise = np.zeros(signal_length)
            n_spikes = np.random.randint(1, 5)
            spike_indices = np.random.choice(signal_length, n_spikes, replace=False)
            noise[spike_indices] = np.random.uniform(-1, 1, n_spikes)

        noisy = clean + noise
        clean_signals.append(clean)
        noisy_signals.append(noisy)

    return np.array(clean_signals, dtype="float32"), np.array(noisy_signals, dtype="float32")



# 2. MSE and SNR Computation

def compute_mse(clean, noisy_or_denoised):
    return np.mean((clean - noisy_or_denoised)**2, axis=1)

def compute_snr(clean, noisy_or_denoised):
    noise_power = np.mean((clean - noisy_or_denoised)**2, axis=1)
    signal_power = np.mean(clean**2, axis=1)
    return 10 * np.log10(signal_power / noise_power)



# 3. Prepare training data

signal_length = 256
clean, noisy = generate_signals(n_samples=2000, signal_length=signal_length)

clean = clean[..., np.newaxis]  # shape: (samples, 256, 1)
noisy = noisy[..., np.newaxis]


# 4. Build 1D CNN Autoencoder

model = models.Sequential([
    layers.Input(shape=(signal_length,1)),
    
    # Encoder
    layers.Conv1D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling1D(2, padding='same'),
    layers.Conv1D(16, 5, padding='same', activation='relu'),
    layers.MaxPooling1D(2, padding='same'),
    
    # Bottleneck
    layers.Conv1D(16, 3, padding='same', activation='relu'),
    
    # Decoder
    layers.UpSampling1D(2),
    layers.Conv1D(32, 5, padding='same', activation='relu'),
    layers.UpSampling1D(2),
    layers.Conv1D(1, 5, padding='same', activation='linear')  # output layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
model.summary()


# 5. Train

model.fit(noisy, clean, epochs=50, batch_size=32, validation_split=0.1, verbose=1)


# 6. Generate test signals

def generate_test_signals(signal_length=256):
    x = np.linspace(0, 4*np.pi, signal_length)
    test_clean = []
    test_noisy = []

    for waveform in ["sine", "square", "sawtooth"]:
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.5, 1.5)

        if waveform == "sine":
            clean = amplitude * np.sin(freq * x + phase)
        elif waveform == "square":
            clean = amplitude * np.sign(np.sin(freq * x + phase))
        else:
            clean = amplitude * (2 * ((freq*x/(2*np.pi) + phase/(2*np.pi)) % 1) - 1)

        # Add random noise
        noise_type = np.random.choice(["gaussian", "uniform", "impulse"])
        if noise_type == "gaussian":
            noise = np.random.normal(0, 0.5, signal_length)
        elif noise_type == "uniform":
            noise = np.random.uniform(-0.5, 0.5, signal_length)
        else:
            noise = np.zeros(signal_length)
            n_spikes = np.random.randint(1, 5)
            spike_indices = np.random.choice(signal_length, n_spikes, replace=False)
            noise[spike_indices] = np.random.uniform(-1, 1, n_spikes)

        noisy = clean + noise
        test_clean.append(clean)
        test_noisy.append(noisy)

    test_clean = np.array(test_clean, dtype='float32')[..., np.newaxis]
    test_noisy = np.array(test_noisy, dtype='float32')[..., np.newaxis]
    return test_clean, test_noisy

test_clean, test_noisy = generate_test_signals(signal_length=signal_length)


# 7. Denoise test signals

denoised_test = model.predict(test_noisy)


# 8. Quantitative Comparison between Raw Noisy Signal and Denoised Output

mse_noisy = compute_mse(test_clean.squeeze(), test_noisy.squeeze())
mse_denoised = compute_mse(test_clean.squeeze(), denoised_test.squeeze())
snr_noisy = compute_snr(test_clean.squeeze(), test_noisy.squeeze())
snr_denoised = compute_snr(test_clean.squeeze(), denoised_test.squeeze())

waveforms = ["Sine", "Square", "Sawtooth"]

for i, waveform in enumerate(waveforms):
    print(f"{waveform} Wave:")
    print(f"  MSE before denoising: {mse_noisy[i]:.4f}")
    print(f"  MSE after denoising:  {mse_denoised[i]:.4f}")
    print(f"  SNR before denoising: {snr_noisy[i]:.2f} dB")
    print(f"  SNR after denoising:  {snr_denoised[i]:.2f} dB\n")


# 9. Plot results

plt.figure(figsize=(15, 8))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(test_clean[i].squeeze(), label="Clean")
    plt.plot(test_noisy[i].squeeze(), label="Noisy", alpha=0.5)
    plt.plot(denoised_test[i].squeeze(), label="Denoised")
    plt.legend()
    plt.title(f"{waveforms[i]} Wave - 1D Denoiser")

plt.tight_layout()
plt.show()
