"""
Data loading and preprocessing utilities.

This module provides functions to load and preprocess geophone and DAS data
from synthetic or real datasets.
"""
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
import glob

logger = logging.getLogger(__name__)

def load_seismic_trace(file_path):
    """
    Load a seismic trace file.
    
    Args:
        file_path (str): Path to the trace file
        
    Returns:
        tuple: (times, amplitude) arrays
    """
    try:
        data = np.loadtxt(file_path)
        times = data[:, 0]
        amplitude = data[:, 1]
        return times, amplitude
    except Exception as e:
        logger.error(f"Error loading trace file {file_path}: {str(e)}")
        return None, None

def load_traces_from_directory(directory, pattern="*.txt"):
    """
    Load all trace files from a directory.
    
    Args:
        directory (str): Directory containing trace files
        pattern (str, optional): File pattern to match
        
    Returns:
        tuple: (times, data) where data has shape (n_channels, n_time_steps)
    """
    directory_path = Path(directory)
    file_paths = sorted(list(directory_path.glob(pattern)))
    
    if not file_paths:
        logger.warning(f"No files found in {directory} matching pattern {pattern}")
        return None, None
    
    # Load the first file to get time samples
    times, first_trace = load_seismic_trace(file_paths[0])
    
    if times is None:
        return None, None
    
    n_time_steps = len(times)
    n_channels = len(file_paths)
    
    # Initialize data array
    data = np.zeros((n_channels, n_time_steps))
    data[0, :] = first_trace  # Add the first trace
    
    # Load remaining traces
    for i, file_path in enumerate(file_paths[1:], start=1):
        _, trace = load_seismic_trace(file_path)
        if trace is not None and len(trace) == n_time_steps:
            data[i, :] = trace
        else:
            logger.warning(f"Skipping {file_path}: incompatible time samples")
            
    return times, data

def load_geophone_and_das_data(geophone_dir, das_dir):
    """
    Load geophone and DAS data from directories.
    
    Args:
        geophone_dir (str): Directory containing geophone trace files
        das_dir (str): Directory containing DAS trace files
        
    Returns:
        tuple: (times, geophone_data, das_data) arrays
    """
    # Load geophone data
    times, geophone_data = load_traces_from_directory(geophone_dir)
    
    if times is None:
        logger.error(f"Failed to load geophone data from {geophone_dir}")
        return None, None, None
    
    # Load DAS data
    das_times, das_data = load_traces_from_directory(das_dir)
    
    if das_times is None:
        logger.error(f"Failed to load DAS data from {das_dir}")
        return times, geophone_data, None
    
    # Check time alignment
    if not np.array_equal(times, das_times):
        logger.warning("Time samples differ between geophone and DAS data")
        # Resample if needed (simplified here)
        if len(times) == len(das_times):
            logger.info("Assuming same sampling rate, aligning time axes")
        else:
            logger.error("Cannot align time axes with different lengths")
            return times, geophone_data, None
    
    return times, geophone_data, das_data

def normalize_data(data, method='max'):
    """
    Normalize seismic data.
    
    Args:
        data (numpy.ndarray): Data array with shape (n_channels, n_time_steps)
        method (str): Normalization method ('max', 'std', or 'minmax')
        
    Returns:
        numpy.ndarray: Normalized data with the same shape
    """
    normalized = np.zeros_like(data)
    
    if method == 'max':
        # Normalize each channel by its maximum absolute value
        for i in range(data.shape[0]):
            max_val = np.max(np.abs(data[i, :]))
            if max_val > 0:
                normalized[i, :] = data[i, :] / max_val
            else:
                normalized[i, :] = data[i, :]
                
    elif method == 'std':
        # Normalize each channel to zero mean and unit standard deviation
        for i in range(data.shape[0]):
            mean = np.mean(data[i, :])
            std = np.std(data[i, :])
            if std > 0:
                normalized[i, :] = (data[i, :] - mean) / std
            else:
                normalized[i, :] = data[i, :] - mean
                
    elif method == 'minmax':
        # Normalize each channel to range [0, 1]
        for i in range(data.shape[0]):
            min_val = np.min(data[i, :])
            max_val = np.max(data[i, :])
            if max_val > min_val:
                normalized[i, :] = (data[i, :] - min_val) / (max_val - min_val)
            else:
                normalized[i, :] = data[i, :]
    
    return normalized

def create_masked_dataset(geophone_data, das_data, mask_pattern='random', mask_ratio=0.3):
    """
    Create a dataset with masked geophone channels for training.
    
    Args:
        geophone_data (numpy.ndarray): Geophone data array with shape (n_channels, n_time_steps)
        das_data (numpy.ndarray): DAS data array with shape (n_das_channels, n_time_steps)
        mask_pattern (str): Masking pattern ('random', 'regular', or 'block')
        mask_ratio (float): Ratio of channels to mask (0.0 to 1.0)
        
    Returns:
        tuple: (masked_geophone_data, mask, target_data)
            masked_geophone_data: Geophone data with masked channels
            mask: Boolean array indicating masked channels (True = masked)
            target_data: Original geophone data (ground truth)
    """
    n_channels, n_time_steps = geophone_data.shape
    n_masked = int(n_channels * mask_ratio)
    
    # Create a copy of the original data as the target
    target_data = geophone_data.copy()
    
    # Initialize mask (False = keep, True = mask)
    mask = np.zeros(n_channels, dtype=bool)
    
    if mask_pattern == 'random':
        # Randomly select channels to mask
        mask_indices = np.random.choice(n_channels, n_masked, replace=False)
        mask[mask_indices] = True
        
    elif mask_pattern == 'regular':
        # Regularly spaced channels
        step = max(1, n_channels // n_masked)
        mask_indices = np.arange(0, n_channels, step)[:n_masked]
        mask[mask_indices] = True
        
    elif mask_pattern == 'block':
        # Contiguous block of channels
        start_idx = np.random.randint(0, n_channels - n_masked + 1)
        mask[start_idx:start_idx+n_masked] = True
    
    # Create masked data by setting masked channels to zeros or special value
    masked_geophone_data = geophone_data.copy()
    masked_geophone_data[mask, :] = 0.0  # Could be NaN or another placeholder
    
    return masked_geophone_data, mask, target_data

def align_geophone_das_spatial(geophone_coords, das_coords, das_data):
    """
    Align DAS channels spatially with geophone locations.
    
    Args:
        geophone_coords (numpy.ndarray): Geophone coordinates, shape (n_geophones, 3)
        das_coords (numpy.ndarray): DAS channel coordinates, shape (n_das_channels, 3)
        das_data (numpy.ndarray): DAS data, shape (n_das_channels, n_time_steps)
        
    Returns:
        numpy.ndarray: DAS data aligned to geophone locations, shape (n_geophones, n_time_steps)
    """
    n_geophones = len(geophone_coords)
    n_time_steps = das_data.shape[1]
    
    # Initialize aligned data
    aligned_das = np.zeros((n_geophones, n_time_steps))
    
    # For each geophone, find the nearest DAS channel(s)
    for i, geo_coord in enumerate(geophone_coords):
        # Compute distances to all DAS channels
        distances = np.linalg.norm(das_coords - geo_coord, axis=1)
        
        # Find the nearest DAS channel
        nearest_idx = np.argmin(distances)
        
        # Simple approach: use the nearest channel
        aligned_das[i, :] = das_data[nearest_idx, :]
        
        # Alternative: could use distance-weighted interpolation of multiple nearby channels
        
    return aligned_das

def prepare_dataset_for_training(times, geophone_data, das_data, window_size=256, stride=64):
    """
    Prepare windowed dataset for model training.
    
    Args:
        times (numpy.ndarray): Time samples array
        geophone_data (numpy.ndarray): Geophone data, shape (n_channels, n_time_steps)
        das_data (numpy.ndarray): DAS data, shape (n_das_channels, n_time_steps)
        window_size (int): Size of time windows
        stride (int): Stride between windows
        
    Returns:
        tuple: (geophone_windows, das_windows) arrays for training
    """
    n_channels, n_time_steps = geophone_data.shape
    n_das_channels = das_data.shape[0]
    
    # Calculate number of windows
    n_windows = (n_time_steps - window_size) // stride + 1
    
    # Initialize window arrays
    geophone_windows = np.zeros((n_windows, n_channels, window_size))
    das_windows = np.zeros((n_windows, n_das_channels, window_size))
    
    # Extract windows
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        geophone_windows[i, :, :] = geophone_data[:, start_idx:end_idx]
        das_windows[i, :, :] = das_data[:, start_idx:end_idx]
    
    return geophone_windows, das_windows

def split_dataset(geophone_windows, das_windows, split_ratio=(0.7, 0.15, 0.15)):
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        geophone_windows (numpy.ndarray): Geophone data windows
        das_windows (numpy.ndarray): DAS data windows
        split_ratio (tuple): Ratios for (train, val, test) splits
        
    Returns:
        tuple: (train_data, val_data, test_data) where each is a tuple of (geophone, das)
    """
    n_samples = len(geophone_windows)
    indices = np.random.permutation(n_samples)
    
    # Calculate split indices
    train_end = int(split_ratio[0] * n_samples)
    val_end = train_end + int(split_ratio[1] * n_samples)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create dataset splits
    train_data = (geophone_windows[train_indices], das_windows[train_indices])
    val_data = (geophone_windows[val_indices], das_windows[val_indices])
    test_data = (geophone_windows[test_indices], das_windows[test_indices])
    
    return train_data, val_data, test_data