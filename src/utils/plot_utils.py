"""
Plotting utilities for seismic data visualization.

This module provides functions to visualize seismic data, model performance,
and evaluation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from matplotlib.animation import FuncAnimation
import logging

logger = logging.getLogger(__name__)

def plot_seismic_trace(times, amplitude, title='Seismic Trace', xlabel='Time (s)', ylabel='Amplitude', 
                      save_path=None, show=True):
    """
    Plot a single seismic trace.
    
    Args:
        times (numpy.ndarray): Time values
        amplitude (numpy.ndarray): Amplitude values
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, amplitude)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_multiple_traces(times, traces, labels=None, title='Seismic Traces', 
                        xlabel='Time (s)', ylabel='Amplitude', save_path=None, show=True):
    """
    Plot multiple seismic traces on the same figure.
    
    Args:
        times (numpy.ndarray): Time values
        traces (list): List of amplitude arrays
        labels (list, optional): List of trace labels
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if labels is None:
        labels = [f"Trace {i+1}" for i in range(len(traces))]
    
    for i, trace in enumerate(traces):
        ax.plot(times, trace, label=labels[i])
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_seismic_gather(data, title='Seismic Gather', xlabel='Time Sample', ylabel='Channel',
                      cmap='seismic', vmin=None, vmax=None, aspect='auto', save_path=None, show=True):
    """
    Plot a seismic gather as an image.
    
    Args:
        data (numpy.ndarray): Seismic data of shape (n_channels, n_time_steps)
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        cmap (str, optional): Colormap
        vmin (float, optional): Minimum value for colormap
        vmax (float, optional): Maximum value for colormap
        aspect (str, optional): Image aspect ratio
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if vmin is None:
        vmin = -np.max(np.abs(data))
    if vmax is None:
        vmax = np.max(np.abs(data))
    
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_comparison_gathers(true_data, pred_data, title='True vs Predicted',
                           cmap='seismic', save_path=None, show=True):
    """
    Plot comparison between true and predicted seismic gathers.
    
    Args:
        true_data (numpy.ndarray): True seismic data of shape (n_channels, n_time_steps)
        pred_data (numpy.ndarray): Predicted seismic data of the same shape
        title (str, optional): Plot title
        cmap (str, optional): Colormap
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determine colormap limits for consistent scale
    vmax = max(np.max(np.abs(true_data)), np.max(np.abs(pred_data)))
    vmin = -vmax
    
    # Plot true data
    im1 = ax1.imshow(true_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title('True Data')
    ax1.set_xlabel('Time Sample')
    ax1.set_ylabel('Channel')
    plt.colorbar(im1, ax=ax1)
    
    # Plot predicted data
    im2 = ax2.imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('Predicted Data')
    ax2.set_xlabel('Time Sample')
    plt.colorbar(im2, ax=ax2)
    
    # Plot difference
    diff = true_data - pred_data
    im3 = ax3.imshow(diff, cmap=cmap, aspect='auto')
    ax3.set_title('Difference')
    ax3.set_xlabel('Time Sample')
    plt.colorbar(im3, ax=ax3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_masked_gather(data, mask, title='Masked Gather', cmap='seismic', 
                     mask_color='white', save_path=None, show=True):
    """
    Plot a seismic gather with masked channels.
    
    Args:
        data (numpy.ndarray): Seismic data of shape (n_channels, n_time_steps)
        mask (numpy.ndarray): Boolean mask for channels (True = masked)
        title (str, optional): Plot title
        cmap (str, optional): Colormap
        mask_color (str, optional): Color for masked channels
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a copy of the data with masked channels set to NaN
    masked_data = data.copy()
    masked_data[mask] = np.nan
    
    # Create custom colormap with masked values shown in a different color
    cmap_with_mask = cm.get_cmap(cmap).copy()
    cmap_with_mask.set_bad(color=mask_color)
    
    vmax = np.max(np.abs(data))
    vmin = -vmax
    
    im = ax.imshow(masked_data, cmap=cmap_with_mask, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Time Sample')
    ax.set_ylabel('Channel')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_training_history(history, title='Training History', save_path=None, show=True):
    """
    Plot training and validation loss history.
    
    Args:
        history (dict): Dictionary with 'train_loss', 'val_loss', and optionally 'learning_rate'
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    if 'learning_rate' in history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title(title)
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title(title)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_wavefield_snapshot(wavefield, title='Wavefield Snapshot', cmap='seismic',
                         vmin=None, vmax=None, save_path=None, show=True):
    """
    Plot a snapshot of a seismic wavefield.
    
    Args:
        wavefield (numpy.ndarray): 2D wavefield data
        title (str, optional): Plot title
        cmap (str, optional): Colormap
        vmin (float, optional): Minimum value for colormap
        vmax (float, optional): Maximum value for colormap
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if vmin is None:
        vmin = -np.max(np.abs(wavefield))
    if vmax is None:
        vmax = np.max(np.abs(wavefield))
    
    im = ax.imshow(wavefield, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def create_wavefield_animation(wavefield_frames, title='Wavefield Animation', cmap='seismic',
                            interval=50, save_path=None, show=True):
    """
    Create an animation of a seismic wavefield.
    
    Args:
        wavefield_frames (list): List of 2D wavefield snapshots
        title (str, optional): Animation title
        cmap (str, optional): Colormap
        interval (int, optional): Delay between frames in milliseconds
        save_path (str, optional): Path to save the animation
        show (bool, optional): Whether to display the animation
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = max([np.max(np.abs(frame)) for frame in wavefield_frames])
    vmin = -vmax
    
    im = ax.imshow(wavefield_frames[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title} - Frame 0")
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    def update(frame):
        im.set_array(wavefield_frames[frame])
        ax.set_title(f"{title} - Frame {frame}")
        return [im]
    
    anim = FuncAnimation(fig, update, frames=len(wavefield_frames), interval=interval, blit=True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer='pillow', fps=1000/interval)
        logger.info(f"Saved animation to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return anim

def plot_metrics_comparison(metrics_list, model_names, title='Model Comparison',
                         save_path=None, show=True):
    """
    Plot comparison of evaluation metrics for different models.
    
    Args:
        metrics_list (list): List of dictionaries containing metrics for each model
        model_names (list): List of model names
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Get all unique metrics
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(list(all_metrics))
    n_metrics = len(all_metrics)
    n_models = len(model_names)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(all_metrics):
        values = []
        for metrics in metrics_list:
            values.append(metrics.get(metric, 0))
        
        axes[i].bar(model_names, values)
        axes[i].set_title(f"{metric}")
        axes[i].grid(True, axis='y')
        
        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def plot_scatter_true_vs_pred(y_true, y_pred, title='True vs Predicted',
                           xlabel='True', ylabel='Predicted', save_path=None, show=True):
    """
    Create scatter plot of true vs predicted values.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot scatter points
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot y=x line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'r-', alpha=0.8, zorder=0)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(0.05, 0.95, f"Correlation: {correlation:.4f}", transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def compare_multiple_models(times, true_data, predictions_list, model_names,
                          channel_idx=0, title='Model Comparison',
                          save_path=None, show=True):
    """
    Compare predictions from multiple models for a single channel.
    
    Args:
        times (numpy.ndarray): Time values
        true_data (numpy.ndarray): True data, shape (n_channels, n_time_steps)
        predictions_list (list): List of predicted data arrays
        model_names (list): List of model names
        channel_idx (int, optional): Channel index to plot
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true data
    ax.plot(times, true_data[channel_idx], 'k-', label='True', linewidth=2)
    
    # Plot predictions from each model
    colors = plt.cm.tab10.colors
    for i, (predictions, model_name) in enumerate(zip(predictions_list, model_names)):
        color = colors[i % len(colors)]
        ax.plot(times, predictions[channel_idx], '-', color=color, label=model_name, alpha=0.7)
    
    ax.set_title(f"{title} - Channel {channel_idx}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig