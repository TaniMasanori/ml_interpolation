#!/usr/bin/env python3
"""
Animate Wavefield from SPECFEM3D Simulation Data

This script creates an animation of seismic wave propagation from SPECFEM3D simulation data.
It visualizes how seismic waves propagate across the receiver array over time.

Features:
- Load seismograms and create animated visualization
- Show all three components (X, Y, Z) side by side
- Customize animation duration, colormap, and normalization
- Save as MP4 or GIF for sharing
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from plot_shot_gather import load_seismograms

def create_wavefield_animation(specfem_dir=None, data_type="velocity", 
                              normalize=True, duration=10, fps=30,
                              cmap="seismic", output_file=None, dpi=150):
    """
    Create an animation of wave propagation from seismogram data.
    
    Args:
        specfem_dir (str): Path to SPECFEM3D installation
        data_type (str): Type of data to load (velocity, displacement, acceleration)
        normalize (bool): Whether to normalize the data
        duration (float): Duration of the animation in seconds
        fps (int): Frames per second
        cmap (str): Colormap to use
        output_file (str): Output file path (must end with .mp4 or .gif)
        dpi (int): DPI for output file
        
    Returns:
        matplotlib.animation.Animation: Animation object
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
    
    # Create figure with three subplots (one for each component)
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])
    
    # Create axes for each component
    axes = []
    images = []
    titles = []
    
    for i, component in enumerate(components):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        
        # Initial empty image
        im = ax.imshow(
            np.zeros((data_arrays[i].shape[0], 1)),
            aspect='auto',
            cmap=cmap,
            vmin=-1 if normalize else -np.max(np.abs(data_arrays[i])),
            vmax=1 if normalize else np.max(np.abs(data_arrays[i])),
            extent=[0, 1, data_arrays[i].shape[0], 1],
            origin='upper'
        )
        images.append(im)
        
        # Add title and labels
        title = ax.set_title(f"{data_type.capitalize()} {component}", fontsize=12)
        titles.append(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Receiver Number')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set y-ticks with station names
        stations = stations_list[i]
        if len(stations) > 20:
            step = len(stations) // 10
            tick_positions = np.arange(1, len(stations) + 1, step)
            tick_labels = [stations[i-1] for i in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_yticks(np.arange(1, len(stations) + 1))
            ax.set_yticklabels(stations)
    
    # Add overall title
    fig.suptitle(f'Seismic Wave Propagation - {data_type.capitalize()}', fontsize=16)
    plt.tight_layout()
    
    # Time display text
    time_text = fig.text(0.5, 0.01, 'Time: 0.00 s', ha='center', fontsize=12)
    
    # Calculate animation parameters
    total_frames = int(duration * fps)
    frame_indices = np.linspace(0, len(times) - 1, total_frames).astype(int)
    
    # Create animation
    def update_frame(frame_idx):
        time_idx = frame_indices[frame_idx]
        current_time = times[time_idx]
        
        # Update time display
        time_text.set_text(f'Time: {current_time:.2f} s')
        
        # Update each component's image
        for i in range(len(components)):
            # Create a window of data centered on the current time
            window_size = int(len(times) / 20)  # 5% of the total time samples
            window_start = max(0, time_idx - window_size // 2)
            window_end = min(len(times) - 1, time_idx + window_size // 2)
            
            # Get data for the window
            window_data = data_arrays[i][:, window_start:window_end+1]
            window_times = times[window_start:window_end+1]
            
            # Update image data and extent
            images[i].set_array(window_data)
            images[i].set_extent([window_times[0], window_times[-1], data_arrays[i].shape[0], 1])
            
            # Update title to show current time
            titles[i].set_text(f"{data_type.capitalize()} {components[i]} - t={current_time:.2f}s")
        
        return images + [time_text] + titles
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update_frame, frames=total_frames, 
        interval=1000/fps, blit=False
    )
    
    # Save animation if output file is specified
    if output_file:
        if output_file.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
            anim.save(output_file, writer=writer, dpi=dpi)
        elif output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        else:
            raise ValueError("Output file must end with .mp4 or .gif")
        
        print(f"Animation saved to {output_file}")
    
    return anim

def main():
    """Main function to parse arguments and create animation."""
    parser = argparse.ArgumentParser(description="Animate wavefield from SPECFEM3D simulation data")
    
    # Path arguments
    parser.add_argument("--specfem-dir", type=str, default="~/specfem3d",
                      help="Path to SPECFEM3D installation directory")
    parser.add_argument("--output-file", type=str, default="wavefield_animation.mp4",
                      help="Output animation file (mp4 or gif)")
    
    # Data selection arguments
    parser.add_argument("--data-type", type=str, default="velocity", 
                      choices=["velocity", "displacement", "acceleration"],
                      help="Data type to plot (velocity, displacement, acceleration)")
    
    # Animation parameters
    parser.add_argument("--duration", type=float, default=10.0,
                      help="Duration of the animation in seconds")
    parser.add_argument("--fps", type=int, default=30,
                      help="Frames per second")
    parser.add_argument("--dpi", type=int, default=150,
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
    
    try:
        # Create animation
        create_wavefield_animation(
            specfem_dir=specfem_dir,
            data_type=args.data_type,
            normalize=args.normalize,
            duration=args.duration,
            fps=args.fps,
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