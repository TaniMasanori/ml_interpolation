#!/usr/bin/env python3
"""
Plot velocity model layers for Vp, Vs, and density.
Based on the model parameters in the JSON file.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_parameter(layers, param_name, title, cmap_name, unit_str, output_file=None):
    """
    Plot a layered model parameter (Vp, Vs, or density)
    
    Args:
        layers (list): List of layer dictionaries
        param_name (str): Parameter to plot ('vp', 'vs', or 'rho')
        title (str): Title for the plot
        cmap_name (str): Colormap name
        unit_str (str): Unit string for annotations
        output_file (str): Output file path
    """
    # Calculate layer depths
    depths = [0]
    thickness_sum = 0
    for layer in layers:
        thickness_sum += layer["thickness"]
        depths.append(thickness_sum)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define distance range (x-axis)
    distance = np.linspace(0, 500, 100)
    
    # Create a custom colormap
    cmap = plt.get_cmap(cmap_name)
    norm = colors.Normalize(
        vmin=min([layer[param_name] for layer in layers]),
        vmax=max([layer[param_name] for layer in layers])
    )
    
    # Plot each layer as a rectangle
    for i, layer in enumerate(layers):
        # Create a filled rectangle for each layer
        bottom = depths[i]
        height = depths[i+1] - depths[i]
        rect = plt.Rectangle((0, bottom), 500, height, 
                             color=cmap(norm(layer[param_name])))
        ax.add_patch(rect)
        
        # Add text annotation in the middle of each layer
        text_x = 250
        text_y = (depths[i] + depths[i+1]) / 2
        ax.text(text_x, text_y, f"{layer[param_name]} {unit_str}", 
                 ha='center', va='center', fontsize=12, 
                 color='black', fontweight='bold')
    
    # Set up axes and labels
    ax.set_xlim(0, 500)
    ax.set_ylim(0, thickness_sum)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (km)')
    ax.set_title(title)
    
    # Invert y-axis for depth to increase downward
    ax.invert_yaxis()
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # You need to set an array for the mappable even if empty
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f'{title} ({unit_str})')
    
    # Save figure if output file specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    plt.tight_layout()
    return fig

def main():
    # Load velocity model from parameter file
    parameter_file = "parameter_sets/updated_velocity_model.json"
    
    if not os.path.exists(parameter_file):
        print(f"Error: Parameter file {parameter_file} not found")
        return
    
    with open(parameter_file, 'r') as f:
        params = json.load(f)
    
    if 'model' not in params or 'layers' not in params['model']:
        print("Error: No velocity model found in parameter file")
        return
    
    layers = params['model']['layers']
    print(f"Loaded {len(layers)} layers from parameter file")
    
    # Create output directory if it doesn't exist
    output_dir = "velocity_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Vp (P-wave velocity)
    plot_parameter(
        layers=layers,
        param_name="vp",
        title="P-wave Velocity Model",
        cmap_name="hot_r",
        unit_str="km/s",
        output_file=os.path.join(output_dir, "vp_model.png")
    )
    
    # Plot Vs (S-wave velocity)
    plot_parameter(
        layers=layers,
        param_name="vs",
        title="S-wave Velocity Model",
        cmap_name="cool",
        unit_str="km/s",
        output_file=os.path.join(output_dir, "vs_model.png")
    )
    
    # Plot density (rho)
    plot_parameter(
        layers=layers,
        param_name="rho",
        title="Density Model",
        cmap_name="viridis",
        unit_str="g/cm³",
        output_file=os.path.join(output_dir, "density_model.png")
    )
    
    print("All plots generated successfully in the 'velocity_plots' directory")

if __name__ == "__main__":
    main() 