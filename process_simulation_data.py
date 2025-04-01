#!/usr/bin/env python3
"""
Process SPECFEM3D simulation data and convert to DAS data.

This script:
1. Reads seismogram files from the SPECFEM3D simulation
2. Processes and organizes the data into numpy arrays
3. Converts geophone data to DAS data using the DAS converter
4. Saves processed data for further analysis
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil
import subprocess
import glob

# Add the project root to path for imports
sys.path.append(os.path.expanduser('~/ml_interpolation'))

# Import project modules
from src.simulation.das_converter import DASConverter
from src.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging(level='INFO')

# Paths
specfem_dir = os.path.expanduser("~/specfem3d")
specfem_output_dir = os.path.join(specfem_dir, "OUTPUT_FILES")
project_root = os.path.expanduser("~/ml_interpolation")
raw_data_dir = os.path.join(project_root, "data/synthetic/raw/simulation1")
processed_geo_dir = os.path.join(project_root, "data/synthetic/processed/simulation1")
processed_das_dir = os.path.join(processed_geo_dir, "das")

# Create output directories
Path(raw_data_dir).mkdir(parents=True, exist_ok=True)
Path(processed_geo_dir).mkdir(parents=True, exist_ok=True)
Path(processed_das_dir).mkdir(parents=True, exist_ok=True)

# DAS parameters
gauge_length = 10.0  # Gauge length in meters
channel_spacing = 10.0  # Spacing between DAS channels in meters

def read_semv_file(file_path):
    """
    Read a SPECFEM3D semv file (velocity seismogram).
    
    Args:
        file_path (str): Path to the .semv file
        
    Returns:
        tuple: (times, velocity_data) arrays
    """
    try:
        data = np.loadtxt(file_path)
        times = data[:, 0]
        velocity = data[:, 1]
        return times, velocity
    except Exception as e:
        logger.error(f"Error reading semv file {file_path}: {str(e)}")
        return None, None

def process_seismograms():
    """
    Process all seismogram files and organize them by component.
    
    Returns:
        dict: Dictionary with processed data arrays
    """
    print("Processing seismogram files...")
    
    # Find all seismogram files for X, Y, Z components
    x_files = sorted(glob.glob(os.path.join(specfem_output_dir, "*.BXX.semv")))
    y_files = sorted(glob.glob(os.path.join(specfem_output_dir, "*.BXY.semv")))
    z_files = sorted(glob.glob(os.path.join(specfem_output_dir, "*.BXZ.semv")))
    
    print(f"Found {len(x_files)} X component files")
    print(f"Found {len(y_files)} Y component files")
    print(f"Found {len(z_files)} Z component files")
    
    if not x_files:
        logger.error("No seismogram files found!")
        return None
    
    # Initialize data arrays
    all_data_x = []
    all_data_y = []
    all_data_z = []
    times = None
    
    # Process X component files
    station_names = []
    for i, file_path in enumerate(x_files):
        station_name = os.path.basename(file_path).split(".")[1]
        station_names.append(station_name)
        
        t, v = read_semv_file(file_path)
        if t is None or v is None:
            continue
            
        if times is None:
            times = t
        elif not np.array_equal(times, t):
            logger.warning(f"Time samples differ in {file_path}, using original time samples")
        
        all_data_x.append(v)
    
    # Process Y component files
    for i, file_path in enumerate(y_files):
        _, v = read_semv_file(file_path)
        if v is not None:
            all_data_y.append(v)
    
    # Process Z component files
    for i, file_path in enumerate(z_files):
        _, v = read_semv_file(file_path)
        if v is not None:
            all_data_z.append(v)
    
    # Convert to numpy arrays
    data_x = np.array(all_data_x)
    data_y = np.array(all_data_y)
    data_z = np.array(all_data_z)
    
    print(f"Processed {data_x.shape[0]} stations, each with {data_x.shape[1]} time samples")
    
    # Create station dataframe
    station_df = pd.DataFrame({
        'name': station_names,
        'x': np.linspace(500, 1500, len(station_names)),
        'y': [1000.0] * len(station_names),
        'z': [0.0] * len(station_names)
    })
    
    # Return all processed data
    return {
        'times': times,
        'data_x': data_x,
        'data_y': data_y,
        'data_z': data_z,
        'station_df': station_df
    }

def save_processed_data(data):
    """
    Save the processed geophone data to the output directory.
    
    Args:
        data (dict): Dictionary with processed data arrays
    """
    print(f"Saving processed data to {processed_geo_dir}...")
    
    # Save numpy arrays
    np.save(os.path.join(processed_geo_dir, "times.npy"), data['times'])
    np.save(os.path.join(processed_geo_dir, "data_x.npy"), data['data_x'])
    np.save(os.path.join(processed_geo_dir, "data_y.npy"), data['data_y'])
    np.save(os.path.join(processed_geo_dir, "data_z.npy"), data['data_z'])
    
    # Save station information
    data['station_df'].to_csv(os.path.join(processed_geo_dir, "stations.csv"), index=False)
    
    print("Geophone data saved successfully")

def convert_to_das(data):
    """
    Convert geophone data to DAS strain rate.
    
    Args:
        data (dict): Dictionary with processed data arrays
        
    Returns:
        numpy.ndarray: DAS strain rate data
    """
    print("Converting geophone data to DAS strain rate...")
    
    # Initialize DAS converter
    das_converter = DASConverter()
    
    # Convert X-component data to DAS strain rate (assuming fiber along X-axis)
    das_data = das_converter.convert_numpy(
        data['data_x'],
        gauge_length=gauge_length,
        channel_spacing=channel_spacing,
        dt=data['times'][1] - data['times'][0]
    )
    
    print(f"Created {das_data.shape[0]} DAS channels, each with {das_data.shape[1]} time samples")
    
    # Save DAS data
    np.save(os.path.join(processed_das_dir, "das_data.npy"), das_data)
    np.save(os.path.join(processed_das_dir, "times.npy"), data['times'])
    
    # Save station information
    data['station_df'].to_csv(os.path.join(processed_das_dir, "stations.csv"), index=False)
    
    print("DAS data saved successfully")
    
    return das_data

def visualize_data(data, das_data):
    """
    Create some visualizations of the geophone and DAS data.
    
    Args:
        data (dict): Dictionary with processed data arrays
        das_data (numpy.ndarray): DAS strain rate data
    """
    print("Creating visualizations...")
    
    # Create output directory for plots
    plots_dir = os.path.join(processed_geo_dir, "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot a single geophone trace
    plt.figure(figsize=(10, 6))
    plt.plot(data['times'], data['data_x'][10])
    plt.title("Geophone X-component - Station 11")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "geophone_trace.png"))
    plt.close()
    
    # Plot a single DAS trace
    plt.figure(figsize=(10, 6))
    plt.plot(data['times'], das_data[10])
    plt.title("DAS Strain Rate - Channel 11")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain Rate")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "das_trace.png"))
    plt.close()
    
    # Plot geophone gather
    plt.figure(figsize=(10, 8))
    plt.imshow(data['data_x'], aspect='auto', cmap='seismic', 
               extent=[data['times'][0], data['times'][-1], 0, data['data_x'].shape[0]])
    plt.colorbar(label="Velocity")
    plt.title("Geophone X-component Gather")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.savefig(os.path.join(plots_dir, "geophone_gather.png"))
    plt.close()
    
    # Plot DAS gather
    plt.figure(figsize=(10, 8))
    plt.imshow(das_data, aspect='auto', cmap='seismic',
               extent=[data['times'][0], data['times'][-1], 0, das_data.shape[0]])
    plt.colorbar(label="Strain Rate")
    plt.title("DAS Strain Rate Gather")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.savefig(os.path.join(plots_dir, "das_gather.png"))
    plt.close()
    
    print(f"Visualizations saved to {plots_dir}")

def main():
    # Step 1: Process SPECFEM3D seismogram files
    processed_data = process_seismograms()
    if processed_data is None:
        print("Error processing seismogram files. Exiting.")
        return
    
    # Step 2: Save processed geophone data
    save_processed_data(processed_data)
    
    # Step 3: Convert to DAS data
    das_data = convert_to_das(processed_data)
    
    # Step 4: Create visualizations
    visualize_data(processed_data, das_data)
    
    print("\nAll processing completed successfully!")
    print(f"Processed geophone data available at: {processed_geo_dir}")
    print(f"Processed DAS data available at: {processed_das_dir}")

if __name__ == "__main__":
    main() 