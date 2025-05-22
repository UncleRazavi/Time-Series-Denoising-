# Time Series Denoising 

This Python script provides several methods for denoising and smoothing 1D time series data. It includes both traditional signal processing techniques and a deep learning-based autoencoder.

## Features

- Moving Average
- Fourier Transform Filtering
- Wavelet Denoising
- Savitzky-Golay Filtering
- Kalman Filtering
- PCA-Based Denoising
- Autoencoder Denoising (LSTM-based)

## Function Descriptions

### moving_average(time_series, window_size)
Applies a simple moving average over the specified window size.

### fourier_filtering(time_series, cutoff)
Performs low-pass filtering using the Fourier Transform. Frequencies above the cutoff are removed.

### wavelet_denoising(time_series, wavelet='db1', level=4)
Uses wavelet decomposition and thresholding to reduce noise in the signal.

### savgol_filtering(time_series, window_length, polyorder)
Applies the Savitzky-Golay filter for smoothing while preserving local features.

### kalman_filtering(time_series)
Uses a basic Kalman filter to estimate the denoised signal.

### pca_denoising(time_series)
Reduces noise using Principal Component Analysis (PCA). Assumes the signal is one-dimensional.

### autoencoder_denoising(time_series)
Uses an LSTM-based autoencoder to reconstruct a denoised version of the input signal.

## Example

The script includes an example that:

- Generates a noisy sine wave
- Applies all denoising methods
- Plots the results using matplotlib


## Notes

- Input must be a 1D NumPy array.
- The autoencoder requires reshaping the input and may take time to train.
- PCA and autoencoder are suitable for exploratory or advanced use.

## License

This code is free to use and modify for educational or research purposes.

