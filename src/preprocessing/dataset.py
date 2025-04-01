"""
Dataset classes for seismic data.

This module provides PyTorch dataset implementations for seismic data,
including geophone and DAS data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class SeismicDataset(Dataset):
    """PyTorch Dataset for seismic data with geophone and DAS inputs."""
    
    def __init__(self, geophone_data, das_data, mask_ratio=0.3, mask_pattern='random', transform=None):
        """
        Initialize SeismicDataset.
        
        Args:
            geophone_data (numpy.ndarray): Geophone data windows, shape (n_samples, n_channels, n_time_steps)
            das_data (numpy.ndarray): DAS data windows, shape (n_samples, n_das_channels, n_time_steps)
            mask_ratio (float): Ratio of geophone channels to mask (0.0 to 1.0)
            mask_pattern (str): Masking pattern ('random', 'regular', or 'block')
            transform (callable, optional): Optional transform to apply to the data
        """
        self.geophone_data = torch.from_numpy(geophone_data).float()
        self.das_data = torch.from_numpy(das_data).float()
        self.mask_ratio = mask_ratio
        self.mask_pattern = mask_pattern
        self.transform = transform
        
        # Validate data shapes
        assert self.geophone_data.shape[0] == self.das_data.shape[0], "Number of samples must match"
        
        self.n_samples = self.geophone_data.shape[0]
        self.n_channels = self.geophone_data.shape[1]
        self.n_das_channels = self.das_data.shape[1]
        self.n_time_steps = self.geophone_data.shape[2]
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def _generate_mask(self, index):
        """
        Generate a mask for geophone channels.
        
        Args:
            index (int): Sample index (used for deterministic masking)
            
        Returns:
            torch.Tensor: Boolean mask for geophone channels (True = masked)
        """
        n_masked = int(self.n_channels * self.mask_ratio)
        mask = torch.zeros(self.n_channels, dtype=torch.bool)
        
        if self.mask_pattern == 'random':
            # Use index as random seed for deterministic results
            rng = np.random.RandomState(index)
            mask_indices = rng.choice(self.n_channels, n_masked, replace=False)
            mask[mask_indices] = True
            
        elif self.mask_pattern == 'regular':
            # Regularly spaced channels
            step = max(1, self.n_channels // n_masked)
            mask_indices = np.arange(0, self.n_channels, step)[:n_masked]
            mask[mask_indices] = True
            
        elif self.mask_pattern == 'block':
            # Contiguous block of channels, position determined by index
            start_idx = index % (self.n_channels - n_masked + 1)
            mask[start_idx:start_idx+n_masked] = True
            
        return mask
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index (int): Sample index
            
        Returns:
            tuple: (masked_geophone, das, mask, target_geophone)
                masked_geophone: Geophone data with masked channels
                das: DAS data
                mask: Boolean mask indicating which channels are masked
                target_geophone: Original (unmasked) geophone data
        """
        # Get the original data
        geophone = self.geophone_data[index]
        das = self.das_data[index]
        
        # Generate mask
        mask = self._generate_mask(index)
        
        # Create masked geophone data
        masked_geophone = geophone.clone()
        masked_geophone[mask] = 0.0  # Could use another value or special token
        
        # Apply transformations if provided
        if self.transform:
            masked_geophone, das, geophone = self.transform(masked_geophone, das, geophone)
            
        return masked_geophone, das, mask, geophone
    
class TransformerSeismicDataset(Dataset):
    """PyTorch Dataset for transformer-based seismic interpolation."""
    
    def __init__(self, geophone_data, das_data, mask_ratio=0.3, mask_pattern='random', 
                 pad_token=-1.0, positional_encoding=True):
        """
        Initialize TransformerSeismicDataset.
        
        Args:
            geophone_data (numpy.ndarray): Geophone data windows, shape (n_samples, n_channels, n_time_steps)
            das_data (numpy.ndarray): DAS data windows, shape (n_samples, n_das_channels, n_time_steps)
            mask_ratio (float): Ratio of geophone channels to mask (0.0 to 1.0)
            mask_pattern (str): Masking pattern ('random', 'regular', or 'block')
            pad_token (float): Value to use for padding tokens
            positional_encoding (bool): Whether to generate positional encodings
        """
        self.geophone_data = torch.from_numpy(geophone_data).float()
        self.das_data = torch.from_numpy(das_data).float()
        self.mask_ratio = mask_ratio
        self.mask_pattern = mask_pattern
        self.pad_token = pad_token
        self.positional_encoding = positional_encoding
        
        # Validate data shapes
        assert self.geophone_data.shape[0] == self.das_data.shape[0], "Number of samples must match"
        
        self.n_samples = self.geophone_data.shape[0]
        self.n_channels = self.geophone_data.shape[1]
        self.n_das_channels = self.das_data.shape[1]
        self.n_time_steps = self.geophone_data.shape[2]
        
        # Pre-compute positional encodings if required
        if self.positional_encoding:
            self.geo_positions = self._generate_positions(self.n_channels)
            self.das_positions = self._generate_positions(self.n_das_channels)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def _generate_positions(self, n_channels):
        """
        Generate positional encodings for the channels.
        
        Args:
            n_channels (int): Number of channels
            
        Returns:
            torch.Tensor: Positional encodings, shape (n_channels,)
        """
        return torch.arange(n_channels).float() / n_channels
    
    def _generate_mask(self, index):
        """Generate mask for geophone channels similar to SeismicDataset."""
        n_masked = int(self.n_channels * self.mask_ratio)
        mask = torch.zeros(self.n_channels, dtype=torch.bool)
        
        if self.mask_pattern == 'random':
            rng = np.random.RandomState(index)
            mask_indices = rng.choice(self.n_channels, n_masked, replace=False)
            mask[mask_indices] = True
        elif self.mask_pattern == 'regular':
            step = max(1, self.n_channels // n_masked)
            mask_indices = np.arange(0, self.n_channels, step)[:n_masked]
            mask[mask_indices] = True
        elif self.mask_pattern == 'block':
            start_idx = index % (self.n_channels - n_masked + 1)
            mask[start_idx:start_idx+n_masked] = True
            
        return mask
    
    def __getitem__(self, index):
        """
        Get a sample prepared for transformer model input.
        
        For transformer processing, we:
        1. Concatenate DAS and geophone data along the channel dimension
        2. Create attention masks for the transformer
        3. Provide positional encodings
        
        Args:
            index (int): Sample index
            
        Returns:
            tuple: (input_data, attention_mask, positions, target)
                input_data: Concatenated [DAS, masked_geophone] data
                attention_mask: Mask for attention (1=attend, 0=ignore)
                positions: Channel positions for positional encoding
                target: Original geophone data (for computing loss)
        """
        # Get the original data
        geophone = self.geophone_data[index]
        das = self.das_data[index]
        
        # Generate mask for geophone channels
        geo_mask = self._generate_mask(index)
        
        # Create masked geophone data
        masked_geophone = geophone.clone()
        masked_geophone[geo_mask] = self.pad_token  # Use pad token for masked channels
        
        # Concatenate DAS and masked geophone data
        # Shape: (n_das_channels + n_geo_channels, n_time_steps)
        input_data = torch.cat([das, masked_geophone], dim=0)
        
        # Create attention mask (1 = attend to this position, 0 = ignore)
        # All DAS channels are attended to
        das_attention = torch.ones(self.n_das_channels, dtype=torch.bool)
        # Only unmasked geophone channels are attended to
        geo_attention = ~geo_mask
        attention_mask = torch.cat([das_attention, geo_attention], dim=0)
        
        # Create positional encodings if required
        if self.positional_encoding:
            positions = torch.cat([self.das_positions, self.geo_positions], dim=0)
        else:
            positions = None
            
        # Return the sample
        return input_data, attention_mask, positions, geophone