"""
Evaluation metrics for seismic interpolation.

This module provides functions to compute various metrics for evaluating
the performance of seismic interpolation models.
"""
import numpy as np
import torch
from scipy.signal import coherence
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def mean_squared_error(y_true, y_pred):
    """
    Compute mean squared error between true and predicted signals.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_channels, n_time_steps) or (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        
    Returns:
        float: Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Compute mean absolute error between true and predicted signals.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_channels, n_time_steps) or (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        
    Returns:
        float: Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))

def signal_to_noise_ratio(y_true, y_pred):
    """
    Compute signal-to-noise ratio between true and predicted signals.
    
    SNR = 10 * log10(var(y_true) / var(y_true - y_pred))
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_channels, n_time_steps) or (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        
    Returns:
        float: Signal-to-noise ratio in dB
    """
    if y_true.ndim == 1:
        signal_var = np.var(y_true)
        noise_var = np.var(y_true - y_pred)
        if noise_var == 0:
            return float('inf')
        return 10 * np.log10(signal_var / noise_var)
    else:
        snrs = []
        for i in range(y_true.shape[0]):
            signal_var = np.var(y_true[i])
            noise_var = np.var(y_true[i] - y_pred[i])
            if noise_var == 0:
                snrs.append(float('inf'))
            else:
                snrs.append(10 * np.log10(signal_var / noise_var))
        return np.mean(snrs)

def correlation_coefficient(y_true, y_pred):
    """
    Compute Pearson correlation coefficient between true and predicted signals.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_channels, n_time_steps) or (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        
    Returns:
        float: Correlation coefficient (mean across channels if multiple)
    """
    if y_true.ndim == 1:
        r, _ = pearsonr(y_true, y_pred)
        return r
    else:
        rs = []
        for i in range(y_true.shape[0]):
            r, _ = pearsonr(y_true[i], y_pred[i])
            rs.append(r)
        return np.mean(rs)

def coherence_measure(y_true, y_pred, fs=100, nperseg=256):
    """
    Compute coherence between true and predicted signals.
    
    Coherence measures the degree of linear dependency of two signals in the frequency domain.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        fs (float): Sampling frequency
        nperseg (int): Length of each FFT segment
        
    Returns:
        tuple: (frequencies, coherence values)
    """
    f, cxy = coherence(y_true, y_pred, fs=fs, nperseg=nperseg)
    return f, cxy

def frequency_domain_error(y_true, y_pred):
    """
    Compute error in frequency domain between true and predicted signals.
    
    This calculates the Mean Squared Error between the amplitude spectra.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        
    Returns:
        float: Mean squared error in frequency domain
    """
    # Compute FFTs
    Y_true = np.abs(np.fft.fft(y_true))
    Y_pred = np.abs(np.fft.fft(y_pred))
    
    # Compute MSE in frequency domain
    freq_mse = np.mean((Y_true - Y_pred) ** 2)
    
    return freq_mse

def amplitude_ratio(y_true, y_pred):
    """
    Compute ratio of peak amplitudes between true and predicted signals.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_channels, n_time_steps) or (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        
    Returns:
        float: Ratio of peak amplitudes (mean across channels if multiple)
    """
    if y_true.ndim == 1:
        true_amp = np.max(np.abs(y_true))
        pred_amp = np.max(np.abs(y_pred))
        if true_amp == 0:
            return float('inf') if pred_amp > 0 else 1.0
        return pred_amp / true_amp
    else:
        ratios = []
        for i in range(y_true.shape[0]):
            true_amp = np.max(np.abs(y_true[i]))
            pred_amp = np.max(np.abs(y_pred[i]))
            if true_amp == 0:
                ratios.append(float('inf') if pred_amp > 0 else 1.0)
            else:
                ratios.append(pred_amp / true_amp)
        return np.mean(ratios)

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on test data.
    
    Args:
        model (torch.nn.Module): Trained model
        test_loader (torch.utils.data.DataLoader): Test data loader
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {
        'mse': [],
        'mae': [],
        'snr': [],
        'corr': [],
        'amp_ratio': []
    }
    
    with torch.no_grad():
        for batch in test_loader:
            # Prepare inputs (this may need to be adjusted based on the specific model input)
            if len(batch) == 4:  # SeismicDataset output
                masked_geophone, das, mask, target_geophone = batch
                masked_geophone = masked_geophone.to(device)
                das = das.to(device)
                mask = mask.to(device)
                target_geophone = target_geophone.to(device)
                
                # Forward pass (adjust this based on your model's forward method)
                outputs = model(das, masked_geophone, mask)
                
                # Extract results for masked channels only
                for i in range(masked_geophone.size(0)):  # Loop over batch
                    batch_mask = mask[i]
                    for j in range(masked_geophone.size(1)):  # Loop over channels
                        if batch_mask[j]:  # If channel is masked
                            true = target_geophone[i, j].cpu().numpy()
                            pred = outputs[i, j].cpu().numpy()
                            
                            # Compute metrics
                            metrics['mse'].append(mean_squared_error(true, pred))
                            metrics['mae'].append(mean_absolute_error(true, pred))
                            metrics['snr'].append(signal_to_noise_ratio(true, pred))
                            metrics['corr'].append(correlation_coefficient(true, pred))
                            metrics['amp_ratio'].append(amplitude_ratio(true, pred))
            
            else:  # Likely TransformerSeismicDataset output
                input_data, attention_mask, positions, target = batch
                input_data = input_data.to(device)
                attention_mask = attention_mask.to(device)
                if positions is not None:
                    positions = positions.to(device)
                target = target.to(device)
                
                # Forward pass for transformer model
                outputs = model(input_data, attention_mask=attention_mask, position_ids=positions)
                
                # Extract only the geophone portion of the outputs
                n_das_channels = outputs.size(1) - target.size(1)
                predicted_geophone = outputs[:, n_das_channels:, :]
                
                # Compare predicted and target geophone data
                for i in range(target.size(0)):  # Loop over batch
                    for j in range(target.size(1)):  # Loop over channels
                        true = target[i, j].cpu().numpy()
                        pred = predicted_geophone[i, j].cpu().numpy()
                        
                        # Compute metrics
                        metrics['mse'].append(mean_squared_error(true, pred))
                        metrics['mae'].append(mean_absolute_error(true, pred))
                        metrics['snr'].append(signal_to_noise_ratio(true, pred))
                        metrics['corr'].append(correlation_coefficient(true, pred))
                        metrics['amp_ratio'].append(amplitude_ratio(true, pred))
    
    # Compute average metrics
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    return avg_metrics

def plot_trace_comparison(y_true, y_pred, times=None, title='Trace Comparison', save_path=None):
    """
    Plot comparison between true and predicted traces.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        times (numpy.ndarray, optional): Time values for x-axis
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if times is None:
        times = np.arange(len(y_true))
    
    ax.plot(times, y_true, 'b-', label='True')
    ax.plot(times, y_pred, 'r-', label='Predicted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Add metrics as text
    mse = mean_squared_error(y_true, y_pred)
    snr = signal_to_noise_ratio(y_true, y_pred)
    corr = correlation_coefficient(y_true, y_pred)
    
    metrics_text = f'MSE: {mse:.6f}\nSNR: {snr:.2f} dB\nCorr: {corr:.4f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Saved trace comparison to {save_path}')
    
    return fig

def plot_gather_comparison(true_gather, pred_gather, mask=None, title='Gather Comparison', save_path=None):
    """
    Plot comparison between true and predicted seismic gathers.
    
    Args:
        true_gather (numpy.ndarray): True gather of shape (n_channels, n_time_steps)
        pred_gather (numpy.ndarray): Predicted gather of the same shape
        mask (numpy.ndarray, optional): Boolean mask indicating masked channels
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Plot true gather
    im0 = axes[0].imshow(true_gather, aspect='auto', cmap='seismic')
    axes[0].set_title('True Gather')
    axes[0].set_xlabel('Time Sample')
    axes[0].set_ylabel('Channel')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot predicted gather
    im1 = axes[1].imshow(pred_gather, aspect='auto', cmap='seismic')
    axes[1].set_title('Predicted Gather')
    axes[1].set_xlabel('Time Sample')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot difference
    diff = true_gather - pred_gather
    im2 = axes[2].imshow(diff, aspect='auto', cmap='seismic')
    axes[2].set_title('Difference')
    axes[2].set_xlabel('Time Sample')
    plt.colorbar(im2, ax=axes[2])
    
    # Mark masked channels if provided
    if mask is not None:
        for i, masked in enumerate(mask):
            if masked:
                axes[0].axhline(i, color='r', linestyle='--', alpha=0.5)
                axes[1].axhline(i, color='r', linestyle='--', alpha=0.5)
                axes[2].axhline(i, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Saved gather comparison to {save_path}')
    
    return fig

def plot_frequency_comparison(y_true, y_pred, fs=100, title='Frequency Domain Comparison', save_path=None):
    """
    Plot frequency domain comparison between true and predicted signals.
    
    Args:
        y_true (numpy.ndarray): True signal of shape (n_time_steps,)
        y_pred (numpy.ndarray): Predicted signal of the same shape
        fs (float): Sampling frequency
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Compute FFTs
    n = len(y_true)
    freq = np.fft.fftfreq(n, d=1/fs)
    Y_true = np.abs(np.fft.fft(y_true))
    Y_pred = np.abs(np.fft.fft(y_pred))
    
    # Plot amplitude spectra
    positive_freq = freq > 0
    ax1.plot(freq[positive_freq], Y_true[positive_freq], 'b-', label='True')
    ax1.plot(freq[positive_freq], Y_pred[positive_freq], 'r-', label='Predicted')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Amplitude Spectrum')
    ax1.set_xlim(0, fs/2)
    ax1.legend()
    ax1.grid(True)
    
    # Plot coherence
    f, cxy = coherence_measure(y_true, y_pred, fs=fs)
    ax2.plot(f, cxy)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Coherence')
    ax2.set_title('Magnitude-Squared Coherence')
    ax2.set_xlim(0, fs/2)
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    # Add frequency domain error as text
    freq_mse = frequency_domain_error(y_true, y_pred)
    ax1.text(0.02, 0.98, f'Freq MSE: {freq_mse:.6f}', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Saved frequency comparison to {save_path}')
    
    return fig