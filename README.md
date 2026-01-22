# 1D-Signal-Denoiser
This project implements a TensorFlow-based denoising autoencoder for 1D signals. It trains on paired clean and noisy data, learns to remove noise, and reconstructs cleaner signals. Supports per-signal normalization and visualization, suitable for audio, sensor, or time-series denoising tasks.

## Motivation

Traditional signal denoising methods rely on handcrafted filters and assumptions about noise statistics. This project explores a data-driven approach using neural network autoencoders to learn signal reconstruction directly from noisy observations.

## Project Highlights:
- Designed and trained a neural network autoencoder for 1D signal denoising
- Generated synthetic sinusoidal datasets with controlled Gaussian noise
- Implemented data preprocessing, training, validation, and inference pipelines
- Visualized model performance using Matplotlib
- Achieved effective noise suppression using Mean Squared Error optimization

## Technical Stack

- **Language:** Python  
- **Frameworks:** TensorFlow / Keras  
- **Libraries:** NumPy, Matplotlib  
- **Techniques:** Autoencoders, Supervised Learning, Regression, Signal Processing

## Model Architecture
Fully connected autoencoder:
Input (256)
 → Dense (128, ReLU)
 → Dense (64, ReLU)
 → Dense (128, ReLU)
 → Dense (256, Linear)

- Optimizer: Adam (learning rate = 1e-3)
- Loss Function: Mean Squared Error (MSE)
- Training: 50 epochs, batch size = 32, 10% validation split

## Dataset

- Synthetic 1D sinusoidal signals with random phase shifts
- Gaussian noise (σ = 0.5) added to clean signals
- Dataset size: 2000 samples
- Signal length: 256 points

## Results

The trained autoencoder successfully reconstructs clean signals from noisy inputs, demonstrating:
- Noise reduction while preserving signal structure
- Generalization to unseen noisy samples

Model performance is evaluated visually by comparing clean, noisy, and denoised signals.

## Future Improvements
- Train on diverse real-world signals and noise types to improve generalization beyond synthetic data.
- Replace fully connected layers with 1D convolutional autoencoders to better capture local temporal signal structure.
