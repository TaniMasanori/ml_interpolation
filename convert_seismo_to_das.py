#!/usr/bin/env python3
"""
Convert SPECFEM3D seismograms to DAS strain data format.

This script takes seismogram files from SPECFEM3D simulation and
converts them to the Distributed Acoustic Sensing (DAS) strain format.
It supports converting individual seismograms or creating a simulated
DAS fiber array from multiple stations.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
import pandas as pd
from scipy.signal import butter, filtfilt

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('das_conversion.log')
    ]
)
logger = logging.getLogger(__name__)

# Import DASConverter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.simulation.das_converter import DASConverter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert SPECFEM3D seismograms to DAS format")
    parser.add_argument("--input-dir", type=str, default="data/synthetic/raw/simulation1",
                       help="Directory containing SPECFEM3D seismogram files")
    parser.add_argument("--output-dir", type=str, default="data/synthetic/processed/simulation1",
                       help="Directory to save DAS files")
    parser.add_argument("--gauge-length", type=float, default=10.0,
                       help="DAS gauge length in meters")
    parser.add_argument("--channel-spacing", type=float, default=10.0,
                       help="Spacing between DAS channels in meters")
    parser.add_argument("--component", type=str, default="BXX",
                       help="Seismogram component to process (BXX, BXY, BXZ)")
    parser.add_argument("--fiber-mode", type=str, default="virtual",
                       choices=["virtual", "individual"],
                       help="DAS fiber mode: 'virtual' for creating fibers connecting stations, "
                           "'individual' for converting each station individually")
    parser.add_argument("--filter", action="store_true",
                       help="Apply bandpass filter to the converted data")
    parser.add_argument("--filter-low", type=float, default=0.5,
                       help="Low frequency cutoff for bandpass filter (Hz)")
    parser.add_argument("--filter-high", type=float, default=25.0,
                       help="High frequency cutoff for bandpass filter (Hz)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots of the conversion")
    return parser.parse_args()

def load_station_coords(stations_file):
    """
    Load station coordinates from STATIONS file.
    
    Args:
        stations_file (str): Path to STATIONS file
        
    Returns:
        pd.DataFrame: DataFrame with station coordinates
    """
    stations = []
    with open(stations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                stations.append({
                    'name': parts[0],
                    'network': parts[1],
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'elev': float(parts[4]),
                    'depth': float(parts[5])
                })
    
    return pd.DataFrame(stations)

def load_seismogram(seismo_dir, station_name, component='BXX'):
    """
    Load seismogram data from SPECFEM3D output.
    
    Args:
        seismo_dir (str): Directory containing seismogram files
        station_name (str): Station name
        component (str): Component to load (BXX, BXY, BXZ)
        
    Returns:
        tuple: (time, displacement)
    """
    # For displacement seismograms, file extension is .semd
    # Components: BXX (displacement in X direction), etc.
    seismo_file = os.path.join(seismo_dir, f"{station_name}.{component}.semd")
    
    if not os.path.exists(seismo_file):
        logger.error(f"Seismogram file not found: {seismo_file}")
        return None, None
    
    # Load seismogram data (skip header line)
    try:
        data = np.loadtxt(seismo_file, skiprows=1)
        time = data[:, 0]
        displacement = data[:, 1]
        return time, displacement
    except Exception as e:
        logger.error(f"Error loading seismogram {seismo_file}: {str(e)}")
        return None, None

def calculate_strain(disp1, disp2, distance):
    """
    Calculate strain between two stations.
    
    Args:
        disp1 (np.ndarray): Displacement at station 1
        disp2 (np.ndarray): Displacement at station 2
        distance (float): Distance between stations
        
    Returns:
        np.ndarray: Strain
    """
    return (disp2 - disp1) / distance

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply butterworth bandpass filter to data.
    
    Args:
        data (np.ndarray): Input data
        lowcut (float): Low frequency cutoff (Hz)
        highcut (float): High frequency cutoff (Hz)
        fs (float): Sampling frequency (Hz)
        order (int): Filter order
        
    Returns:
        np.ndarray: Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def find_seismogram_dir(simulation_dir):
    """Find the directory containing seismogram files."""
    # Check if OUTPUT_FILES directory exists
    output_files_dir = os.path.join(simulation_dir, "OUTPUT_FILES")
    if os.path.exists(output_files_dir):
        return output_files_dir
    
    # Otherwise check the simulation dir directly
    if os.path.exists(simulation_dir):
        return simulation_dir
    
    logger.error(f"Could not find seismogram directory in {simulation_dir}")
    return None

def create_virtual_fibers(stations_df, seismograms, time, das_dir, component, 
                          apply_filter=False, filter_low=0.5, filter_high=25.0):
    """
    Create virtual fiber DAS data by calculating strain between adjacent stations.
    
    Args:
        stations_df (pd.DataFrame): DataFrame with station coordinates
        seismograms (dict): Dictionary of seismograms by station name
        time (np.ndarray): Time vector for seismograms
        das_dir (str): Directory to save DAS data
        component (str): Seismogram component being processed
        apply_filter (bool): Whether to apply bandpass filter
        filter_low (float): Low frequency cutoff for filter (Hz)
        filter_high (float): High frequency cutoff for filter (Hz)
        
    Returns:
        list: Metadata for all DAS channels
    """
    logger.info("Creating virtual fiber DAS data...")
    
    # Calculate sampling frequency
    dt = time[1] - time[0]
    fs = 1.0 / dt
    
    # Create horizontal and vertical fiber channels
    nrows = stations_df['y'].nunique()
    ncols = stations_df['x'].nunique()
    
    if nrows < 2 or ncols < 2:
        logger.error(f"Need at least 2 rows and 2 columns for strain calculation, got {nrows} rows and {ncols} columns")
        return []
    
    logger.info(f"Creating virtual fibers for {nrows} rows and {ncols} columns")
    
    # Sort stations by coordinates
    stations_df = stations_df.sort_values(['y', 'x'])
    
    # Create metadata list
    metadata = []
    
    # Create horizontal fiber channels
    logger.info("Creating horizontal fiber channels...")
    for row in range(nrows):
        row_stations = stations_df[stations_df['y'] == stations_df['y'].unique()[row]]
        row_stations = row_stations.sort_values('x')
        
        for i in range(len(row_stations) - 1):
            station1 = row_stations.iloc[i]
            station2 = row_stations.iloc[i+1]
            
            # Get seismograms
            s1_name, s2_name = station1['name'], station2['name']
            if s1_name not in seismograms or s2_name not in seismograms:
                continue
                
            # Calculate distance and midpoint
            distance = np.sqrt((station2['x'] - station1['x'])**2 + (station2['y'] - station1['y'])**2)
            midpoint_x = (station1['x'] + station2['x']) / 2
            midpoint_y = (station1['y'] + station2['y']) / 2
            
            # Calculate strain
            disp1 = seismograms[s1_name][1]
            disp2 = seismograms[s2_name][1]
            strain = calculate_strain(disp1, disp2, distance)
            
            # Apply bandpass filter if requested
            if apply_filter:
                strain = butter_bandpass_filter(strain, filter_low, filter_high, fs)
            
            # Generate channel ID
            channel_id = f"H{row+1}_{i+1}_{component}"
            
            # Save strain data in binary format for efficiency
            strain_file = os.path.join(das_dir, f"{channel_id}.npy")
            np.save(strain_file, strain)
            
            # Add metadata
            metadata.append({
                'channel_id': channel_id,
                'fiber': f"H{row+1}",
                'x': midpoint_x,
                'y': midpoint_y,
                'station1': s1_name,
                'station2': s2_name,
                'distance': distance,
                'component': component,
                'file': os.path.basename(strain_file)
            })
    
    # Create vertical fiber channels
    logger.info("Creating vertical fiber channels...")
    for col in range(ncols):
        col_stations = stations_df[stations_df['x'] == stations_df['x'].unique()[col]]
        col_stations = col_stations.sort_values('y')
        
        for i in range(len(col_stations) - 1):
            station1 = col_stations.iloc[i]
            station2 = col_stations.iloc[i+1]
            
            # Get seismograms
            s1_name, s2_name = station1['name'], station2['name']
            if s1_name not in seismograms or s2_name not in seismograms:
                continue
                
            # Calculate distance and midpoint
            distance = np.sqrt((station2['x'] - station1['x'])**2 + (station2['y'] - station1['y'])**2)
            midpoint_x = (station1['x'] + station2['x']) / 2
            midpoint_y = (station1['y'] + station2['y']) / 2
            
            # Calculate strain
            disp1 = seismograms[s1_name][1]
            disp2 = seismograms[s2_name][1]
            strain = calculate_strain(disp1, disp2, distance)
            
            # Apply bandpass filter if requested
            if apply_filter:
                strain = butter_bandpass_filter(strain, filter_low, filter_high, fs)
            
            # Generate channel ID
            channel_id = f"V{col+1}_{i+1}_{component}"
            
            # Save strain data in binary format for efficiency
            strain_file = os.path.join(das_dir, f"{channel_id}.npy")
            np.save(strain_file, strain)
            
            # Add metadata
            metadata.append({
                'channel_id': channel_id,
                'fiber': f"V{col+1}",
                'x': midpoint_x,
                'y': midpoint_y,
                'station1': s1_name,
                'station2': s2_name,
                'distance': distance,
                'component': component,
                'file': os.path.basename(strain_file)
            })
    
    logger.info(f"Created {len(metadata)} DAS channels")
    return metadata

def convert_individual_stations(stations_df, seismo_dir, das_dir, gauge_length, channel_spacing, component):
    """
    Convert individual station seismograms to DAS format using the DAS converter.
    
    Args:
        stations_df (pd.DataFrame): DataFrame with station coordinates
        seismo_dir (str): Directory containing seismogram files
        das_dir (str): Directory to save DAS data
        gauge_length (float): DAS gauge length in meters
        channel_spacing (float): Spacing between DAS channels in meters
        component (str): Seismogram component to process
        
    Returns:
        list: Metadata for all DAS channels
    """
    logger.info("Converting individual station seismograms to DAS format...")
    
    # Initialize DAS converter
    converter = DASConverter()
    
    # For individual station conversion, use file pattern to match component
    file_pattern = f"*.{component}.semd"
    
    # Convert files
    num_converted = converter.convert_specfem_directory(
        seismo_dir,
        das_dir,
        gauge_length,
        channel_spacing,
        file_pattern
    )
    
    logger.info(f"Converted {num_converted} files to DAS format")
    
    # Create metadata entries for the converted files
    metadata = []
    das_files = list(Path(das_dir).glob("das_*.txt"))
    
    for das_file in das_files:
        # Extract station name from file name
        # Format is usually: das_STATION.BXX.semd.txt
        parts = das_file.name.split('.')
        if len(parts) < 2:
            continue
            
        # Try to extract station name
        station_name = parts[0].replace("das_", "")
        
        # Find station in stations_df
        station_match = stations_df[stations_df['name'] == station_name]
        if len(station_match) == 0:
            logger.warning(f"Could not find station {station_name} in stations_df")
            continue
            
        station = station_match.iloc[0]
        
        metadata.append({
            'channel_id': f"DAS_{station_name}_{component}",
            'fiber': "individual",
            'x': station['x'],
            'y': station['y'],
            'station': station_name,
            'component': component,
            'gauge_length': gauge_length,
            'channel_spacing': channel_spacing,
            'file': das_file.name
        })
    
    logger.info(f"Created metadata for {len(metadata)} DAS channels")
    return metadata

def plot_station_map(stations_df, metadata, output_dir):
    """
    Generate a plot showing station locations and DAS channels.
    
    Args:
        stations_df (pd.DataFrame): DataFrame with station coordinates
        metadata (list): List of DAS channel metadata
        output_dir (str): Directory to save plot
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot station locations
    plt.scatter(stations_df['x'], stations_df['y'], marker='s', s=50, color='black')
    for _, station in stations_df.iterrows():
        plt.text(station['x'], station['y'], station['name'], fontsize=8)
    
    # Plot fiber connections if using virtual fiber mode
    for item in metadata:
        if 'station1' in item and 'station2' in item:
            # Get station coordinates
            s1 = stations_df[stations_df['name'] == item['station1']].iloc[0]
            s2 = stations_df[stations_df['name'] == item['station2']].iloc[0]
            
            # Plot line connecting stations
            plt.plot([s1['x'], s2['x']], [s1['y'], s2['y']], 'b-', alpha=0.5)
            
            # Plot midpoint (DAS channel location) with different color based on fiber orientation
            if item['fiber'].startswith('H'):
                plt.plot(item['x'], item['y'], 'ro', alpha=0.7, markersize=4)
            else:
                plt.plot(item['x'], item['y'], 'go', alpha=0.7, markersize=4)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Station Map with DAS Channels')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "station_map.png"), dpi=300)
    logger.info(f"Saved station map to {os.path.join(plots_dir, 'station_map.png')}")

def plot_das_examples(metadata, das_dir, time_data, output_dir):
    """
    Generate plots showing examples of DAS data.
    
    Args:
        metadata (list): List of DAS channel metadata
        das_dir (str): Directory containing DAS data
        time_data (np.ndarray): Time vector for data
        output_dir (str): Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot sample of DAS waveforms
    n_samples = min(6, len(metadata))
    
    fig, axs = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    
    for i in range(n_samples):
        item = metadata[i]
        
        # Load DAS data
        file_path = os.path.join(das_dir, item['file'])
        das_data = np.load(file_path)
        
        # Plot data
        if n_samples > 1:
            ax = axs[i]
        else:
            ax = axs
            
        ax.plot(time_data, das_data)
        title = f"DAS Channel: {item['channel_id']}"
        if 'station1' in item and 'station2' in item:
            title += f" (Stations: {item['station1']} - {item['station2']})"
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "das_samples.png"), dpi=300)
    logger.info(f"Saved DAS samples to {os.path.join(plots_dir, 'das_samples.png')}")
    
    # Create DAS gather plot for a single fiber if we have enough data
    horizontal_fibers = set([item['fiber'] for item in metadata if item['fiber'].startswith('H')])
    vertical_fibers = set([item['fiber'] for item in metadata if item['fiber'].startswith('V')])
    
    # Plot one horizontal and one vertical fiber if available
    if horizontal_fibers:
        plot_fiber_gather(horizontal_fibers.pop(), metadata, das_dir, time_data, plots_dir, "horizontal")
    
    if vertical_fibers:
        plot_fiber_gather(vertical_fibers.pop(), metadata, das_dir, time_data, plots_dir, "vertical")

def plot_fiber_gather(fiber_id, metadata, das_dir, time_data, plots_dir, fiber_type):
    """
    Generate a gather plot for a single fiber.
    
    Args:
        fiber_id (str): ID of the fiber to plot
        metadata (list): List of DAS channel metadata
        das_dir (str): Directory containing DAS data
        time_data (np.ndarray): Time vector for data
        plots_dir (str): Directory to save plots
        fiber_type (str): Type of fiber (horizontal or vertical)
    """
    # Get all channels for this fiber
    fiber_channels = [item for item in metadata if item['fiber'] == fiber_id]
    
    if len(fiber_channels) < 2:
        return
    
    # Sort channels based on position (x for horizontal, y for vertical)
    if fiber_type == "horizontal":
        fiber_channels.sort(key=lambda x: x['x'])
    else:
        fiber_channels.sort(key=lambda x: x['y'])
    
    # Load all data
    gather_data = []
    for channel in fiber_channels:
        file_path = os.path.join(das_dir, channel['file'])
        das_data = np.load(file_path)
        gather_data.append(das_data)
    
    # Convert to numpy array
    gather_array = np.array(gather_data)
    
    # Create gather plot
    plt.figure(figsize=(10, 8))
    
    # Calculate max amplitude for consistent scaling
    vmax = np.percentile(np.abs(gather_array), 98)
    
    plt.imshow(gather_array, aspect='auto', cmap='seismic',
              extent=[time_data[0], time_data[-1], len(gather_data), 1],
              vmin=-vmax, vmax=vmax)
    
    plt.colorbar(label='Strain')
    plt.title(f"DAS Gather - {fiber_id} Fiber")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    
    # Add channel labels
    channel_labels = [channel['channel_id'] for channel in fiber_channels]
    plt.yticks(np.arange(1.5, len(channel_labels) + 0.5), channel_labels, fontsize=8)
    
    plt.savefig(os.path.join(plots_dir, f"das_gather_{fiber_id}.png"), dpi=300)
    logger.info(f"Saved DAS gather plot to {os.path.join(plots_dir, f'das_gather_{fiber_id}.png')}")

def main():
    """Convert SPECFEM3D seismograms to DAS format."""
    args = parse_args()
    
    # Expand paths
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    
    logger.info(f"Converting seismograms from {input_dir} to DAS format")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Component: {args.component}")
    logger.info(f"Fiber mode: {args.fiber_mode}")
    if args.fiber_mode == "individual":
        logger.info(f"DAS parameters: gauge length={args.gauge_length}m, channel spacing={args.channel_spacing}m")
    if args.filter:
        logger.info(f"Applying bandpass filter: {args.filter_low}-{args.filter_high} Hz")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    das_dir = os.path.join(output_dir, "das")
    os.makedirs(das_dir, exist_ok=True)
    
    # Find seismogram directory
    seismo_dir = find_seismogram_dir(input_dir)
    if not seismo_dir:
        logger.error("Could not find seismogram directory. Exiting.")
        return None
    
    # Load station coordinates
    stations_file = os.path.join(input_dir, "STATIONS")
    if not os.path.exists(stations_file):
        logger.error(f"STATIONS file not found: {stations_file}")
        return None
    
    stations_df = load_station_coords(stations_file)
    logger.info(f"Loaded {len(stations_df)} stations from {stations_file}")
    
    # Save station coordinates to output directory
    stations_df.to_csv(os.path.join(output_dir, "stations.csv"), index=False)
    
    # Process based on fiber mode
    if args.fiber_mode == "virtual":
        # Load seismograms for all stations
        logger.info("Loading seismograms for all stations...")
        seismograms = {}
        for _, station in stations_df.iterrows():
            station_name = station['name']
            time, displacement = load_seismogram(seismo_dir, station_name, args.component)
            if time is not None:
                seismograms[station_name] = (time, displacement)
        
        if not seismograms:
            logger.error("No seismograms loaded")
            return None
        
        # Get time vector from the first seismogram
        station_name = list(seismograms.keys())[0]
        time_data = seismograms[station_name][0]
        
        # Save time vector
        np.save(os.path.join(das_dir, "time.npy"), time_data)
        
        # Create virtual fibers and get metadata
        metadata = create_virtual_fibers(
            stations_df, 
            seismograms, 
            time_data, 
            das_dir, 
            args.component,
            apply_filter=args.filter, 
            filter_low=args.filter_low, 
            filter_high=args.filter_high
        )
    else:  # individual mode
        # Convert individual stations
        metadata = convert_individual_stations(
            stations_df,
            seismo_dir,
            das_dir,
            args.gauge_length,
            args.channel_spacing,
            args.component
        )
        
        # If using individual mode, we need to load time data from one of the output files
        if metadata:
            das_file = os.path.join(das_dir, metadata[0]['file'])
            if os.path.exists(das_file):
                # For text format output from DASConverter
                data = np.loadtxt(das_file)
                time_data = data[:, 0]
                np.save(os.path.join(das_dir, "time.npy"), time_data)
            else:
                logger.error(f"Could not find DAS file: {das_file}")
                time_data = None
        else:
            logger.error("No DAS data was generated")
            return None
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(das_dir, "metadata.csv"), index=False)
    logger.info(f"Saved metadata to {os.path.join(das_dir, 'metadata.csv')}")
    
    # Generate plots if requested
    if args.plot and metadata:
        logger.info("Generating plots...")
        plot_station_map(stations_df, metadata, output_dir)
        
        if time_data is not None:
            plot_das_examples(metadata, das_dir, time_data, output_dir)
    
    logger.info(f"DAS conversion completed. Output saved to {output_dir}")
    print(f"\n=== DAS Conversion Completed Successfully ===")
    print(f"Output saved to: {output_dir}")
    print(f"- DAS data: {das_dir}")
    print(f"- Station coordinates: {os.path.join(output_dir, 'stations.csv')}")
    print(f"- DAS metadata: {os.path.join(das_dir, 'metadata.csv')}")
    if args.plot:
        print(f"- Preview plots: {os.path.join(output_dir, 'plots')}")
    print("\nNext steps:")
    print("1. Process data: python notebooks/03_data_preprocessing.ipynb")
    print("2. Train model: python notebooks/04_model_training.ipynb")
    return output_dir

if __name__ == "__main__":
    main()