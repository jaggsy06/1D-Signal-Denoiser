import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# 1. Generate synthetic signals

def generate_signals(n_samples=2000, signal_length=256):
    x = np.linspace(0, 4 * np.pi, signal_length)
    clean = np.array([
        np.sin(x + np.random.rand() * 2 * np.pi)
        for _ in range(n_samples)
    ])
    noise = np.random.normal(0, 0.5, clean.shape)
    noisy = clean + noise
    return clean, noisy

clean, noisy = generate_signals()

# 2. Normalize
clean = clean.astype("float32")
noisy = noisy.astype("float32")


# 3. Build Autoencoder

input_dim = clean.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(input_dim)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()


# 4. Train

model.fit(
    noisy,
    clean,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 5. Denoise a signal (Test)

denoised = model.predict(noisy[:1])


# Plot results

plt.figure(figsize=(10, 4))
plt.plot(clean[0], label="Clean")
plt.plot(noisy[0], label="Noisy", alpha=0.5)
plt.plot(denoised[0], label="Denoised")
plt.legend()
plt.title("TensorFlow Signal Denoiser")
plt.show()


