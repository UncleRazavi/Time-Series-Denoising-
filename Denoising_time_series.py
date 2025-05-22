# Moving Average
def moving_average(time_series, window_size):
    import pandas as pd
    return pd.Series(time_series).rolling(window=window_size).mean()

# Fourier Transform Filtering
def fourier_filtering(time_series, cutoff):
    from scipy.fft import fft, ifft
    fft_data = fft(time_series)
    frequencies = np.fft.fftfreq(len(fft_data))
    filtered_fft = fft_data * (abs(frequencies) < cutoff)
    return np.real(ifft(filtered_fft))

# Wavelet Denoising
def wavelet_denoising(time_series, wavelet='db1', level=4):
    import pywt
    coeffs = pywt.wavedec(time_series, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(time_series)))
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    return pywt.waverec(coeffs_thresholded, wavelet)

# Savitzky-Golay Filter
def savgol_filtering(time_series, window_length, polyorder):
    from scipy.signal import savgol_filter
    return savgol_filter(time_series, window_length, polyorder)

# Kalman Filtering
def kalman_filtering(time_series):
    from pykalman import KalmanFilter
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, _ = kf.filter(time_series)
    return state_means.flatten()
# Principal Component Analysis (PCA)
def pca_denoising(time_series):
    from sklearn.decomposition import PCA
    time_series_reshaped = time_series.reshape(-1, 1)
    pca = PCA(n_components=1)
    return pca.inverse_transform(pca.fit_transform(time_series_reshaped)).flatten()

# Deep Learning-based Denoising (Autoencoder)
def autoencoder_denoising(time_series):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    scaler = MinMaxScaler()
    time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1))
    X = np.array([time_series_scaled[i:i+10] for i in range(len(time_series_scaled)-10)])
    
    model = Sequential([
        LSTM(16, activation='relu', input_shape=(10, 1), return_sequences=False),
        RepeatVector(10),
        LSTM(16, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=50, batch_size=16, verbose=0)
    
    denoised_signal = model.predict(X).flatten()
    return scaler.inverse_transform(denoised_signal.reshape(-1, 1)).flatten()

# Example Usage
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate example noisy sine wave
    time_series = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))

    # Apply each denoising method
    ma_result = moving_average(time_series, window_size=5)
    fft_result = fourier_filtering(time_series, cutoff=0.1)
    wavelet_result = wavelet_denoising(time_series)
    sg_result = savgol_filtering(time_series, window_length=11, polyorder=2)
    kalman_result = kalman_filtering(time_series)
    pca_result = pca_denoising(time_series)

    # Plot results
    plt.figure(figsize=(15, 10))
    plt.plot(time_series, label="Original", alpha=0.6)
    plt.plot(ma_result, label="Moving Average")
    plt.plot(fft_result, label="Fourier Filtering")
    plt.plot(wavelet_result, label="Wavelet Denoising")
    plt.plot(sg_result, label="Savitzky-Golay")
    plt.plot(kalman_result, label="Kalman Filtering")
    plt.legend()
    plt.show()
