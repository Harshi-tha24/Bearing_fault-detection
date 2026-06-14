"""
Improvement 2: Advanced Signal Denoising Utilities
===================================================

Three complementary denoising methods:
1. Wavelet Denoising (VisuShrink) - Powerful for bearing signals
2. Savitzky-Golay Filter - Smooth while preserving edges
3. FFT-based Filtering - Clean frequency domain approach

Panel Value:
"We employ multi-method denoising to improve fault feature clarity
under real-world noisy industrial conditions, increasing robustness."

Usage:
    from improvement_2_signal_denoising import denoise_signal
    
    clean_signal = denoise_signal(
        noisy_signal, 
        method='wavelet',  # 'wavelet', 'savgol', or 'fft'
        sr=16000
    )
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.fft import fft, ifft, fftfreq
import pywt
import matplotlib.pyplot as plt


class SignalDenoiser:
    """
    Multi-method signal denoising for bearing vibration signals
    """
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    # =========================================================================
    # METHOD 1: WAVELET DENOISING (VisuShrink + BayesShrink)
    # =========================================================================
    
    def denoise_wavelet(self, signal, wavelet='db4', level=6, method='visuShrink'):
        """
        Wavelet denoising using VisuShrink or BayesShrink
        
        Parameters:
        -----------
        signal : 1D array
            Input noisy signal
        wavelet : str
            Wavelet family ('db4', 'db8', 'sym5', etc.)
        level : int
            Decomposition level (higher = more detail preservation)
        method : str
            'visuShrink' or 'bayesShrink'
        
        Returns:
        --------
        denoised : 1D array
            Denoised signal
        """
        # Decompose
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Estimate noise standard deviation (Donoho method)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        if method == 'visuShrink':
            # Universal threshold
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            
            # Apply soft thresholding
            denoised_coeffs = [coeffs[0]]  # cA (approximation)
            for coeff in coeffs[1:]:
                shrunk = self._soft_threshold(coeff, threshold)
                denoised_coeffs.append(shrunk)
        
        elif method == 'bayesShrink':
            # BayesShrink thresholds
            denoised_coeffs = [coeffs[0]]
            for coeff in coeffs[1:]:
                # Bayesian threshold
                var_coeff = np.var(coeff)
                threshold = (sigma ** 2) / np.sqrt(max(var_coeff - sigma**2, 0))
                shrunk = self._soft_threshold(coeff, threshold)
                denoised_coeffs.append(shrunk)
        
        # Reconstruct
        denoised = pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]
        
        return denoised
    
    @staticmethod
    def _soft_threshold(coeff, threshold):
        """Soft thresholding operator"""
        return np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)
    
    # =========================================================================
    # METHOD 2: SAVITZKY-GOLAY FILTER (Preserves edges & peaks)
    # =========================================================================
    
    def denoise_savgol(self, signal, window_length=11, polyorder=3):
        """
        Savitzky-Golay filter - excellent for bearing signals
        
        Preserves sharp transitions (faults) while smoothing noise
        
        Parameters:
        -----------
        signal : 1D array
            Input signal
        window_length : int
            Smoothing window (must be odd, typically 11-51)
        polyorder : int
            Polynomial order (typically 2-5)
        
        Returns:
        --------
        denoised : 1D array
            Filtered signal
        """
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        denoised = savgol_filter(signal, window_length, polyorder)
        return denoised
    
    # =========================================================================
    # METHOD 3: FFT-BASED FILTERING
    # =========================================================================
    
    def denoise_fft(self, signal, cutoff_freq=5000, filter_type='lowpass'):
        """
        Frequency domain denoising
        
        Parameters:
        -----------
        signal : 1D array
            Input signal
        cutoff_freq : float
            Cutoff frequency (Hz)
        filter_type : str
            'lowpass', 'highpass', or 'bandpass'
        
        Returns:
        --------
        denoised : 1D array
            Filtered signal in time domain
        """
        # FFT
        fft_signal = fft(signal)
        freqs = fftfreq(len(signal), 1/self.sr)
        
        # Create frequency mask
        if filter_type == 'lowpass':
            mask = np.abs(freqs) <= cutoff_freq
        elif filter_type == 'highpass':
            mask = np.abs(freqs) >= cutoff_freq
        elif filter_type == 'bandpass':
            # For bandpass, cutoff_freq is a tuple (low, high)
            mask = (np.abs(freqs) >= cutoff_freq[0]) & (np.abs(freqs) <= cutoff_freq[1])
        
        # Apply mask
        fft_filtered = fft_signal * mask
        
        # Inverse FFT
        denoised = np.real(ifft(fft_filtered))
        
        return denoised
    
    # =========================================================================
    # ENSEMBLE DENOISING (Combine multiple methods)
    # =========================================================================
    
    def denoise_ensemble(self, signal, weights=None):
        """
        Ensemble denoising - combine all three methods
        
        Provides robustness through method diversity
        """
        if weights is None:
            weights = [0.4, 0.35, 0.25]  # Wavelet, Savgol, FFT
        
        # Apply all methods
        wavelet_denoised = self.denoise_wavelet(signal, method='visuShrink')
        savgol_denoised = self.denoise_savgol(signal)
        fft_denoised = self.denoise_fft(signal)
        
        # Weighted ensemble
        ensemble = (
            weights[0] * wavelet_denoised +
            weights[1] * savgol_denoised +
            weights[2] * fft_denoised
        )
        
        return ensemble


def denoise_signal(signal, method='wavelet', **kwargs):
    """
    Convenience function for signal denoising
    
    Parameters:
    -----------
    signal : 1D array
        Input signal
    method : str
        'wavelet', 'savgol', 'fft', or 'ensemble'
    **kwargs : dict
        Method-specific parameters
    
    Returns:
    --------
    denoised : 1D array
        Denoised signal
    """
    denoiser = SignalDenoiser(sr=kwargs.get('sr', 16000))
    
    if method == 'wavelet':
        return denoiser.denoise_wavelet(
            signal,
            wavelet=kwargs.get('wavelet', 'db4'),
            level=kwargs.get('level', 6),
            method=kwargs.get('method', 'visuShrink')
        )
    elif method == 'savgol':
        return denoiser.denoise_savgol(
            signal,
            window_length=kwargs.get('window_length', 11),
            polyorder=kwargs.get('polyorder', 3)
        )
    elif method == 'fft':
        return denoiser.denoise_fft(
            signal,
            cutoff_freq=kwargs.get('cutoff_freq', 5000),
            filter_type=kwargs.get('filter_type', 'lowpass')
        )
    elif method == 'ensemble':
        return denoiser.denoise_ensemble(
            signal,
            weights=kwargs.get('weights', None)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_denoising_methods(signal, noise_signal, sr=16000):
    """
    Compare all denoising methods on a signal
    
    Returns visualization and metrics
    """
    denoiser = SignalDenoiser(sr=sr)
    
    # Apply methods
    wavelet_denoised = denoiser.denoise_wavelet(noise_signal)
    savgol_denoised = denoiser.denoise_savgol(noise_signal)
    fft_denoised = denoiser.denoise_fft(noise_signal)
    ensemble_denoised = denoiser.denoise_ensemble(noise_signal)
    
    # Compute metrics
    def mse(estimated, true):
        return np.mean((estimated - true) ** 2)
    
    def snr_db(signal, noise):
        return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    
    results = {
        'Noisy': {
            'signal': noise_signal,
            'mse': mse(noise_signal, signal),
            'snr': snr_db(signal, noise_signal - signal)
        },
        'Wavelet': {
            'signal': wavelet_denoised,
            'mse': mse(wavelet_denoised, signal)
        },
        'Savgol': {
            'signal': savgol_denoised,
            'mse': mse(savgol_denoised, signal)
        },
        'FFT': {
            'signal': fft_denoised,
            'mse': mse(fft_denoised, signal)
        },
        'Ensemble': {
            'signal': ensemble_denoised,
            'mse': mse(ensemble_denoised, signal)
        }
    }
    
    # Visualize
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    t = np.arange(len(signal)) / sr
    
    # Original signal
    axes[0, 0].plot(t, signal, linewidth=1, color='black', label='Clean Signal')
    axes[0, 0].set_title('Original Signal', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Noisy signal
    axes[0, 1].plot(t, noise_signal, linewidth=0.8, alpha=0.7, label='Noisy')
    axes[0, 1].plot(t, signal, linewidth=1, color='black', alpha=0.5, label='True Signal')
    axes[0, 1].set_title('Noisy Signal', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Denoised methods
    methods = [('Wavelet', wavelet_denoised), ('Savgol', savgol_denoised), 
               ('FFT', fft_denoised), ('Ensemble', ensemble_denoised)]
    
    for idx, (name, denoised) in enumerate(methods):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        
        axes[row, col].plot(t, denoised, linewidth=1, alpha=0.8, label=f'{name} (Denoised)')
        axes[row, col].plot(t, signal, linewidth=1, color='black', alpha=0.3, label='True Signal')
        mse_val = results[name]['mse']
        axes[row, col].set_title(f'{name}\nMSE: {mse_val:.6f}', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('denoising_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Denoising comparison saved: denoising_comparison.png")
    plt.close()
    
    return results


if __name__ == "__main__":
    print("\n🔧 Signal Denoising Utilities for Bearing Fault Detection")
    print("="*80)
    
    # Create synthetic signal with noise
    np.random.seed(42)
    t = np.linspace(0, 1, 2048)
    
    # True signal (bearing fault pattern)
    true_signal = 2 * np.sin(2 * np.pi * 10 * t) + 1.5 * np.sin(2 * np.pi * 50 * t)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.5, len(t))
    noisy_signal = true_signal + noise
    
    print(f"\n📊 Test Signal:")
    print(f"   Length: {len(noisy_signal)} samples")
    print(f"   SNR (noisy): {10 * np.log10(np.mean(true_signal**2) / np.mean(noise**2)):.2f} dB")
    
    # Compare methods
    print("\n🧪 Comparing denoising methods...")
    results = compare_denoising_methods(true_signal, noisy_signal)
    
    print("\n📈 Denoising Results (MSE):")
    print(f"   Noisy Signal: {results['Noisy']['mse']:.6f}")
    print(f"   Wavelet:      {results['Wavelet']['mse']:.6f}")
    print(f"   Savgol:       {results['Savgol']['mse']:.6f}")
    print(f"   FFT:          {results['FFT']['mse']:.6f}")
    print(f"   Ensemble:     {results['Ensemble']['mse']:.6f}")
    
    print("\n" + "="*80)
    print("✅ IMPROVEMENT 2 - Signal Denoising: READY FOR DEPLOYMENT")
    print("="*80)
    print("\n📌 Panel Value:")
    print("   'We employ advanced denoising techniques (wavelet, Savitzky-Goyal, FFT)")
    print("    to improve fault feature clarity under noisy industrial conditions.'")
    print("\n💡 Key Methods:")
    print("   - Wavelet: Excellent for transient fault impulses")
    print("   - Savgol: Preserves sharp edges while smoothing noise")
    print("   - FFT: Clean frequency domain filtering")
    print("   - Ensemble: Combines all three for robustness")
