#!/usr/bin/env python3
"""
Plot Shot Gather from SPECFEM3D Simulation Data

This script generates shot gather visualizations from SPECFEM3D simulation results,
allowing for quick assessment of simulation quality and wave propagation.

Features:
- Load seismograms from SPECFEM3D OUTPUT_FILES directory
- Create shot gather plots (receiver vs. time)
- Support for different components (X, Y, Z)
- Apply various normalization and scaling options
- Save high-resolution images for publication/presentations
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import argparse

def load_seismograms(specfem_dir=None, component="X", data_type="velocity"):
    """
    Load seismogram data from SPECFEM3D output.
    
    Args:
        specfem_dir (str): Path to SPECFEM3D installation
        component (str): Component to load (X, Y, Z)
        data_type (str): Type of data to load (velocity, displacement, acceleration)
    
    Returns:
        tuple: (times, data, stations) where data has shape [n_stations, n_time_samples]
    """
    # Setup paths
    if specfem_dir is None:
        specfem_dir = os.path.expanduser("~/specfem3d")
    
    output_dir = os.path.join(specfem_dir, "OUTPUT_FILES")
    
    # Map data_type to file extension
    ext_map = {
        "velocity": "semv",
        "displacement": "semd",
        "acceleration": "sema"
    }
    
    if data_type not in ext_map:
        raise ValueError(f"Unknown data type: {data_type}. Must be one of {list(ext_map.keys())}")
    
    file_ext = ext_map[data_type]
    
    # Find all seismogram files for the specified component
    pattern = f"*.BX{component}.{file_ext}"
    seismogram_files = sorted(glob.glob(os.path.join(output_dir, pattern)))
    
    if not seismogram_files:
        raise FileNotFoundError(f"No seismogram files found matching pattern {pattern} in {output_dir}")
    
    print(f"Found {len(seismogram_files)} seismogram files for component {component}")
    
    # Load data from each file
    data_list = []
    times = None
    station_names = []
    
    for file_path in seismogram_files:
        # Extract station name from filename
        filename = os.path.basename(file_path)
        station_name = filename.split(".")[1]  # Format is typically NETWORK.STATION.COMPONENT.EXT
        station_names.append(station_name)
        
        # Load data
        try:
            data = np.loadtxt(file_path)
            if times is None:
                times = data[:, 0]
            
            # Append seismogram data (column 1)
            data_list.append(data[:, 1])
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    # Convert to numpy array [n_stations, n_time_samples]
    data_array = np.array(data_list)
    
    return times, data_array, station_names

def create_shot_gather(times, data, stations=None, title=None, normalize=True, clip=1.0, 
                      cmap="seismic", output_file=None, figsize=(10, 8), dpi=300):
    """
    Create a shot gather visualization.
    
    Args:
        times (numpy.ndarray): Time values
        data (numpy.ndarray): Seismogram data with shape [n_stations, n_time_samples]
        stations (list): List of station names
        title (str): Plot title
        normalize (bool): Whether to normalize the data
        clip (float): Clip factor for amplitude (0-1)
        cmap (str): Colormap to use
        output_file (str): Output file path (if None, display the plot)
        figsize (tuple): Figure size
        dpi (int): DPI for output file
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for visualization
    if normalize:
        # Normalize each trace
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            max_val = np.max(np.abs(data[i, :]))
            if max_val > 0:
                normalized_data[i, :] = data[i, :] / max_val
        plot_data = normalized_data
    else:
        # Use raw data
        plot_data = data
    
    # Apply clipping if requested
    if clip < 1.0:
        max_val = np.max(np.abs(plot_data))
        clip_val = max_val * clip
        plot_data = np.clip(plot_data, -clip_val, clip_val)
    
    # Plot the image
    extent = [times[0], times[-1], data.shape[0], 1]
    im = ax.imshow(plot_data, aspect='auto', cmap=cmap, 
                  extent=extent, interpolation='nearest', 
                  origin='upper')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    if normalize:
        cbar.set_label('Normalized Amplitude')
    else:
        cbar.set_label('Amplitude')
    
    # Set labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Receiver Number')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Shot Gather')
    
    # Customize y-axis with station names if provided
    if stations:
        # Show a subset of station names if there are many
        if len(stations) > 20:
            step = len(stations) // 10
            tick_positions = np.arange(1, len(stations) + 1, step)
            tick_labels = [stations[i-1] for i in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_yticks(np.arange(1, len(stations) + 1))
            ax.set_yticklabels(stations)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Shot gather saved to {output_file}")
    
    return fig

def plot_wiggle_traces(times, data, stations=None, title=None, normalize=True, scale=1.0, 
                     output_file=None, figsize=(10, 8), dpi=300):
    """
    Create a wiggle trace plot of seismic data.
    
    Args:
        times (numpy.ndarray): Time values
        data (numpy.ndarray): Seismogram data with shape [n_stations, n_time_samples]
        stations (list): List of station names
        title (str): Plot title
        normalize (bool): Whether to normalize the data
        scale (float): Scaling factor for trace amplitudes
        output_file (str): Output file path (if None, display the plot)
        figsize (tuple): Figure size
        dpi (int): DPI for output file
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        # Normalize each trace
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            max_val = np.max(np.abs(data[i, :]))
            if max_val > 0:
                normalized_data[i, :] = data[i, :] / max_val
        plot_data = normalized_data
    else:
        plot_data = data
    
    # Calculate spacing between traces
    num_traces = plot_data.shape[0]
    
    # Calculate adaptive scaling
    adaptive_scale = scale * (num_traces / 50.0)
    
    # Plot each trace
    for i in range(num_traces):
        trace = plot_data[i, :] * adaptive_scale
        trace_position = num_traces - i  # Invert order to match imshow
        ax.plot(times, trace + trace_position, 'k-', linewidth=0.5)
        
        # Fill positive areas
        ax.fill_between(times, trace_position, trace + trace_position, 
                      where=(trace > 0), interpolate=True, color='black', alpha=0.5)
    
    # Set labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trace Number')
    
    # Set y limits
    ax.set_ylim(0, num_traces + 1)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Wiggle Trace Plot')
        
    # Customize y-axis with station names if provided
    if stations:
        # Show a subset of station names if there are many
        if len(stations) > 20:
            step = len(stations) // 10
            tick_positions = np.arange(1, len(stations) + 1, step)
            tick_labels = [stations[i-1] for i in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_yticks(np.arange(1, len(stations) + 1))
            ax.set_yticklabels(stations)
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Wiggle plot saved to {output_file}")
    
    return fig

def create_combined_plot(times, data, stations=None, component="X", data_type="velocity", 
                       normalize=True, output_dir=None, prefix="shot_gather"):
    """
    Create combined shot gather and wiggle plot.
    
    Args:
        times (numpy.ndarray): Time values
        data (numpy.ndarray): Seismogram data with shape [n_stations, n_time_samples]
        stations (list): List of station names
        component (str): Component name (X, Y, Z)
        data_type (str): Type of data (velocity, displacement, acceleration)
        normalize (bool): Whether to normalize the data
        output_dir (str): Output directory
        prefix (str): Prefix for output file names
    
    Returns:
        tuple: (shot_gather_fig, wiggle_fig)
    """
    # Create title with data information
    title = f"{data_type.capitalize()} Component {component} - {len(stations)} Stations"
    
    # Create output paths
    if output_dir:
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create file paths
        shot_file = os.path.join(output_dir, f"{prefix}_{component}_{data_type}.png")
        wiggle_file = os.path.join(output_dir, f"{prefix}_wiggle_{component}_{data_type}.png")
    else:
        shot_file = None
        wiggle_file = None
    
    # Create shot gather
    shot_fig = create_shot_gather(
        times, data, stations=stations, title=title, 
        normalize=normalize, output_file=shot_file
    )
    
    # Create wiggle plot
    wiggle_fig = plot_wiggle_traces(
        times, data, stations=stations, title=title, 
        normalize=normalize, output_file=wiggle_file
    )
    
    return shot_fig, wiggle_fig

def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(description="Plot shot gather from SPECFEM3D simulation data")
    
    # Path arguments
    parser.add_argument("--specfem-dir", type=str, default="~/specfem3d",
                      help="Path to SPECFEM3D installation directory")
    parser.add_argument("--output-dir", type=str, default="./plots",
                      help="Output directory for plots")
    
    # Data selection arguments
    parser.add_argument("--component", type=str, default="X", choices=["X", "Y", "Z"],
                      help="Seismogram component to plot (X, Y, Z)")
    parser.add_argument("--data-type", type=str, default="velocity", 
                      choices=["velocity", "displacement", "acceleration"],
                      help="Data type to plot (velocity, displacement, acceleration)")
    
    # Plot customization
    parser.add_argument("--normalize", action="store_true", default=True,
                      help="Normalize traces")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                      help="Don't normalize traces")
    parser.add_argument("--clip", type=float, default=1.0,
                      help="Clip factor for amplitude (0-1)")
    parser.add_argument("--cmap", type=str, default="seismic",
                      help="Colormap for shot gather")
    parser.add_argument("--prefix", type=str, default="shot_gather",
                      help="Prefix for output file names")
    
    args = parser.parse_args()
    
    # Expand paths
    specfem_dir = os.path.expanduser(args.specfem_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    try:
        # Load seismogram data
        times, data, stations = load_seismograms(
            specfem_dir=specfem_dir,
            component=args.component,
            data_type=args.data_type
        )
        
        # Create plots
        create_combined_plot(
            times, data, stations=stations,
            component=args.component,
            data_type=args.data_type,
            normalize=args.normalize,
            output_dir=output_dir,
            prefix=args.prefix
        )
        
        print(f"Plots saved to {output_dir}")
        
        # Also create a plot showing all three components if the current component is X
        if args.component == "X":
            print("Creating plots for all components...")
            for comp in ["X", "Y", "Z"]:
                if comp == "X":
                    # Already created this one
                    continue
                
                # Load component data
                try:
                    comp_times, comp_data, comp_stations = load_seismograms(
                        specfem_dir=specfem_dir,
                        component=comp,
                        data_type=args.data_type
                    )
                    
                    # Create plots for this component
                    create_combined_plot(
                        comp_times, comp_data, stations=comp_stations,
                        component=comp,
                        data_type=args.data_type,
                        normalize=args.normalize,
                        output_dir=output_dir,
                        prefix=args.prefix
                    )
                except Exception as e:
                    print(f"Error creating plots for component {comp}: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 