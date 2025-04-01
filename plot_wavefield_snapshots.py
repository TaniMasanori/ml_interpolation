#!/usr/bin/env python3
"""
Plot Wavefield Snapshots from SPECFEM3D Simulation Data

This script generates a grid of snapshots showing the wavefield at different time points.
This is an alternative to animation that doesn't require ffmpeg.

Features:
- Load seismograms and create snapshot visualizations
- Show all three components (X, Y, Z) side by side
- Display multiple time snapshots in a grid layout
- Save high-resolution images for publication/presentations
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
from plot_shot_gather import load_seismograms

def generate_wavefield_snapshots(specfem_dir=None, data_type="velocity", 
                               normalize=True, n_snapshots=9, time_range=None,
                               cmap="seismic", output_file=None, dpi=300,
                               figsize=(12, 10)):
    """
    Generate snapshots of the wavefield at different time points.
    
    Args:
        specfem_dir (str): Path to SPECFEM3D installation
        data_type (str): Type of data to load (velocity, displacement, acceleration)
        normalize (bool): Whether to normalize the data
        n_snapshots (int): Number of snapshots to generate
        time_range (tuple): (start_time, end_time) in seconds, or None for full range
        cmap (str): Colormap to use
        output_file (str): Output file path
        dpi (int): DPI for output file
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Load data for all three components
    components = ["X", "Y", "Z"]
    data_arrays = []
    stations_list = []
    times = None
    
    for component in components:
        try:
            comp_times, comp_data, comp_stations = load_seismograms(
                specfem_dir=specfem_dir,
                component=component,
                data_type=data_type
            )
            
            if times is None:
                times = comp_times
            
            # Normalize if requested
            if normalize:
                # Normalize each trace
                normalized_data = np.zeros_like(comp_data)
                for i in range(comp_data.shape[0]):
                    max_val = np.max(np.abs(comp_data[i, :]))
                    if max_val > 0:
                        normalized_data[i, :] = comp_data[i, :] / max_val
                data_arrays.append(normalized_data)
            else:
                data_arrays.append(comp_data)
                
            stations_list.append(comp_stations)
        except Exception as e:
            print(f"Error loading {component} component: {e}")
            # Use an empty array as a placeholder
            if times is not None:
                empty_data = np.zeros((len(stations_list[0]), len(times)))
                data_arrays.append(empty_data)
                stations_list.append([f"ST{i:03d}" for i in range(len(stations_list[0]))])
    
    if times is None:
        raise ValueError("No data could be loaded for any component")
    
    # Determine time indices for snapshots
    if time_range:
        start_time, end_time = time_range
        start_idx = np.argmin(np.abs(times - start_time))
        end_idx = np.argmin(np.abs(times - end_time))
    else:
        # Skip the first 10% and last 10% of times
        n_times = len(times)
        start_idx = int(0.1 * n_times)
        end_idx = int(0.9 * n_times)
    
    # Calculate snapshot indices
    snapshot_indices = np.linspace(start_idx, end_idx, n_snapshots, dtype=int)
    
    # Calculate grid dimensions for snapshots
    n_rows = int(np.ceil(np.sqrt(n_snapshots)))
    n_cols = int(np.ceil(n_snapshots / n_rows))
    
    # Create figure and main grid
    fig = plt.figure(figsize=figsize)
    
    # Create a main grid with 3 columns (one for each component)
    main_grid = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.3)
    
    # For each component, create a nested grid of snapshots
    for comp_idx, component in enumerate(components):
        # Get the main subplot for this component
        comp_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows + 1, n_cols, 
            subplot_spec=main_grid[0, comp_idx],
            wspace=0.1, hspace=0.2
        )
        
        # Add a title for this component
        title_ax = fig.add_subplot(comp_gs[0, :])
        title_ax.set_title(f"{data_type.capitalize()} Component {component}", fontsize=14)
        title_ax.axis('off')
        
        # Common colorbar limits for this component
        vmin = -1 if normalize else -np.max(np.abs(data_arrays[comp_idx]))
        vmax = 1 if normalize else np.max(np.abs(data_arrays[comp_idx]))
        
        # Create snapshots for this component
        for i, time_idx in enumerate(snapshot_indices):
            if i >= n_snapshots:
                break
                
            # Calculate grid position
            row = (i // n_cols) + 1  # +1 because row 0 is for the title
            col = i % n_cols
            
            # Create axis
            ax = fig.add_subplot(comp_gs[row, col])
            
            # Extract data for this time point (with a small window)
            window_size = 50  # Sample points to each side
            window_start = max(0, time_idx - window_size)
            window_end = min(len(times) - 1, time_idx + window_size)
            window_data = data_arrays[comp_idx][:, window_start:window_end+1]
            window_times = times[window_start:window_end+1]
            
            # Plot the data
            im = ax.imshow(window_data, aspect='auto', cmap=cmap, 
                          extent=[window_times[0], window_times[-1], window_data.shape[0], 1],
                          vmin=vmin, vmax=vmax, origin='upper')
            
            # Add time label
            current_time = times[time_idx]
            ax.set_title(f"t = {current_time:.2f}s", fontsize=10)
            
            # Only add labels on edge plots
            if col == 0:
                ax.set_ylabel('Receiver')
            else:
                ax.set_yticks([])
                
            if row == n_rows:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticks([])

            # Make colorbar for the last plot in each component
            if i == len(snapshot_indices) - 1 or i == n_snapshots - 1:
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                                  pad=0.2, shrink=0.8)
                if normalize:
                    cbar.set_label('Normalized Amplitude', fontsize=8)
                else:
                    cbar.set_label('Amplitude', fontsize=8)
    
    # Add overall title
    plt.suptitle(f'Seismic Wave Propagation Snapshots - {data_type.capitalize()}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Wavefield snapshots saved to {output_file}")
    
    return fig

def main():
    """Main function to parse arguments and generate snapshots."""
    parser = argparse.ArgumentParser(description="Generate wavefield snapshots from SPECFEM3D simulation data")
    
    # Path arguments
    parser.add_argument("--specfem-dir", type=str, default="~/specfem3d",
                      help="Path to SPECFEM3D installation directory")
    parser.add_argument("--output-file", type=str, default="wavefield_snapshots.png",
                      help="Output image file (png, jpg, pdf, etc.)")
    
    # Data selection arguments
    parser.add_argument("--data-type", type=str, default="velocity", 
                      choices=["velocity", "displacement", "acceleration"],
                      help="Data type to plot (velocity, displacement, acceleration)")
    
    # Snapshot parameters
    parser.add_argument("--n-snapshots", type=int, default=9,
                      help="Number of snapshots to generate")
    parser.add_argument("--start-time", type=float, default=None,
                      help="Start time for snapshots (seconds)")
    parser.add_argument("--end-time", type=float, default=None,
                      help="End time for snapshots (seconds)")
    parser.add_argument("--dpi", type=int, default=300,
                      help="DPI for output file")
    
    # Plot customization
    parser.add_argument("--normalize", action="store_true", default=True,
                      help="Normalize traces")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                      help="Don't normalize traces")
    parser.add_argument("--cmap", type=str, default="seismic",
                      help="Colormap for visualization")
    
    args = parser.parse_args()
    
    # Expand paths
    specfem_dir = os.path.expanduser(args.specfem_dir)
    output_file = os.path.expanduser(args.output_file)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Set time range if provided
    time_range = None
    if args.start_time is not None and args.end_time is not None:
        time_range = (args.start_time, args.end_time)
    
    try:
        # Generate snapshots
        generate_wavefield_snapshots(
            specfem_dir=specfem_dir,
            data_type=args.data_type,
            normalize=args.normalize,
            n_snapshots=args.n_snapshots,
            time_range=time_range,
            cmap=args.cmap,
            output_file=output_file,
            dpi=args.dpi
        )
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 