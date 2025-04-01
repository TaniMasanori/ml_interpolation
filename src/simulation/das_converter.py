"""
DAS Converter module.

This module provides functionality to convert SPECFEM3D particle velocity outputs
to Distributed Acoustic Sensing (DAS) strain-rate responses.
"""
import numpy as np
import logging
from pathlib import Path
import subprocess
import os

logger = logging.getLogger(__name__)

class DASConverter:
    """Class to handle conversion from particle velocity to DAS strain rate."""
    
    def __init__(self, genericcode_path=None):
        """
        Initialize DAS converter.
        
        Args:
            genericcode_path (str, optional): Path to the Genericcode tool executable.
                If None, assumes built-in numpy-based conversion methods will be used.
        """
        self.genericcode_path = genericcode_path
        
    def convert_using_genericcode(self, velocity_file, output_file, gauge_length, channel_spacing, config=None):
        """
        Convert velocity data to DAS strain rate using the Genericcode tool.
        
        Args:
            velocity_file (str): Path to velocity data file
            output_file (str): Path to save DAS data
            gauge_length (float): Gauge length in meters
            channel_spacing (float): Spacing between DAS channels in meters
            config (dict, optional): Additional configuration parameters for Genericcode
            
        Returns:
            bool: True if conversion is successful
        """
        if not self.genericcode_path:
            logger.error("Genericcode path not provided. Cannot run external conversion.")
            return False
            
        cmd = [
            self.genericcode_path,
            "-i", velocity_file,
            "-o", output_file,
            "--gauge-length", str(gauge_length),
            "--channel-spacing", str(channel_spacing)
        ]
        
        # Add any additional configuration parameters
        if config:
            for key, value in config.items():
                cmd.extend([f"--{key}", str(value)])
                
        try:
            logger.info(f"Running Genericcode: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("DAS conversion completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"DAS conversion failed: {e.stderr.decode()}")
            return False
            
    def convert_numpy(self, velocity_data, gauge_length, channel_spacing, dt=None):
        """
        Convert velocity data to DAS strain rate using NumPy computations.
        
        This computes the spatial derivative of velocity along the fiber direction,
        which approximates the strain rate measured by DAS.
        
        Args:
            velocity_data (numpy.ndarray): Array of velocity data, shape (n_channels, n_time_steps)
            gauge_length (float): Gauge length in meters
            channel_spacing (float): Spacing between channels in meters
            dt (float, optional): Time step in seconds. If provided, properly scales output
            
        Returns:
            numpy.ndarray: DAS strain rate data, shape (n_channels, n_time_steps)
        """
        # Number of channels to use in the gauge length
        gauge_channels = max(1, int(gauge_length / channel_spacing))
        
        # Pad the velocity data to handle edge effects
        padded_velocity = np.pad(velocity_data, ((gauge_channels, gauge_channels), (0, 0)), mode='edge')
        
        # Initialize output array
        n_channels, n_time_steps = velocity_data.shape
        das_data = np.zeros((n_channels, n_time_steps))
        
        # Compute strain rate using central difference over gauge length
        for i in range(n_channels):
            # Index in padded array
            idx = i + gauge_channels
            # Forward point (half a gauge length ahead)
            forward_idx = idx + gauge_channels // 2
            # Backward point (half a gauge length behind)
            backward_idx = idx - gauge_channels // 2
            
            # Compute spatial derivative over the gauge length
            das_data[i, :] = (padded_velocity[forward_idx, :] - padded_velocity[backward_idx, :]) / gauge_length
            
        # If time step is provided, scale accordingly
        if dt is not None:
            # For strain rate, we would divide by dt, but since we're dealing with
            # spatial derivatives of velocity, not displacement, we don't need this step
            pass
            
        return das_data
    
    def load_specfem_trace(self, trace_file):
        """
        Load a SPECFEM3D trace file and return the particle velocity data.
        
        Args:
            trace_file (str): Path to SPECFEM3D trace file
            
        Returns:
            tuple: (times, velocity_data) arrays
        """
        # Assuming the trace file is in a simple columnar format
        # with time in column 1 and velocity in column 2
        try:
            data = np.loadtxt(trace_file)
            times = data[:, 0]
            velocity = data[:, 1]
            return times, velocity
        except Exception as e:
            logger.error(f"Error loading trace file {trace_file}: {str(e)}")
            return None, None
            
    def convert_specfem_directory(self, input_dir, output_dir, gauge_length, channel_spacing, file_pattern="*.semd"):
        """
        Convert all SPECFEM3D trace files in a directory to DAS strain rate.
        
        Args:
            input_dir (str): Directory containing SPECFEM3D trace files
            output_dir (str): Directory to save DAS data files
            gauge_length (float): Gauge length in meters
            channel_spacing (float): Spacing between channels in meters
            file_pattern (str, optional): Pattern to match trace files
            
        Returns:
            int: Number of files successfully converted
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all trace files
        trace_files = list(input_path.glob(file_pattern))
        
        if not trace_files:
            logger.warning(f"No trace files found in {input_dir} matching pattern {file_pattern}")
            return 0
            
        # Sort files to ensure channels are in order (assuming filename contains channel index)
        trace_files.sort()
        
        # Load all trace data
        all_velocity_data = []
        times = None
        
        for trace_file in trace_files:
            t, v = self.load_specfem_trace(str(trace_file))
            if t is None or v is None:
                continue
                
            if times is None:
                times = t
            elif not np.array_equal(times, t):
                logger.warning(f"Time samples differ in {trace_file}, skipping")
                continue
                
            all_velocity_data.append(v)
            
        if not all_velocity_data:
            logger.error("No valid trace data found")
            return 0
            
        # Convert to array, shape (n_channels, n_time_steps)
        velocity_data = np.array(all_velocity_data)
        
        # Convert to DAS strain rate
        das_data = self.convert_numpy(velocity_data, gauge_length, channel_spacing)
        
        # Save DAS data for each channel
        for i, trace_file in enumerate(trace_files[:len(das_data)]):
            output_file = output_path / f"das_{trace_file.stem}.txt"
            
            # Save as time, strain_rate columns
            output_data = np.column_stack((times, das_data[i, :]))
            np.savetxt(str(output_file), output_data, fmt='%.6e')
            
        logger.info(f"Converted {len(das_data)} trace files to DAS strain rate")
        return len(das_data)