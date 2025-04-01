#!/usr/bin/env python3
"""
Plot a simplified two-layer velocity model similar to the example figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simple_model():
    """Create a simple two-layer velocity model plot."""
    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Set up distance and depth ranges
    x_range = [0, 500]
    depth_range = [0, 250]
    
    # Layer boundaries
    layer1_depth = 100  # depth of first layer boundary
    
    # Layer velocities
    v1 = 1500  # m/s - top layer
    v2 = 5000  # m/s - bottom layer
    
    # Fill the layers with colors
    ax.fill_between([0, 500], [0, 0], [layer1_depth, layer1_depth], color='orange', alpha=0.8)
    ax.fill_between([0, 500], [layer1_depth, layer1_depth], [depth_range[1], depth_range[1]], color='red', alpha=0.8)
    
    # Add text annotations
    ax.text(250, 50, f"{v1} m/s", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(250, 175, f"{v2} m/s", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Plot layer boundaries
    ax.plot([0, 500], [layer1_depth, layer1_depth], 'k-', linewidth=2)
    
    # Add a point source
    ax.plot(150, 0, 'ko', markersize=8, markerfacecolor='black')
    
    # Set up axes
    ax.set_xlim(x_range)
    ax.set_ylim(depth_range)
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Velocity Model', fontsize=14)
    
    # Invert y-axis for depth to increase downward
    ax.invert_yaxis()
    
    # Set grid lines
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure
    output_dir = "velocity_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "simple_velocity_model.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {output_file}")
    
    plt.tight_layout()
    return fig

def main():
    # Create output directory if it doesn't exist
    output_dir = "velocity_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the simple model that exactly matches the provided example
    plot_simple_model()
    
    # Create a two-layer model for each parameter (Vp, Vs, and rho)
    # Define the two-layer model (similar to example)
    two_layer_model = [
        {
            "thickness": 0.1,  # 100 meters in km
            "vp": 1.5,         # 1500 m/s in km/s
            "vs": 0.8,         # 800 m/s in km/s  
            "rho": 1.8         # density in g/cm³
        },
        {
            "thickness": 0.15, # 150 meters in km
            "vp": 5.0,         # 5000 m/s in km/s
            "vs": 2.9,         # 2900 m/s in km/s
            "rho": 2.5         # density in g/cm³
        }
    ]
    
    # Now create individual parameter plots
    try:
        # Import plot_parameter function from the other script
        from plot_velocity_model import plot_parameter
        
        # Generate specific plots for Vp, Vs and density
        for param, title, cmap, unit in [
            ("vp", "P-wave Velocity Model", "hot_r", "km/s"),
            ("vs", "S-wave Velocity Model", "cool", "km/s"),
            ("rho", "Density Model", "viridis", "g/cm³")
        ]:
            # Create the plot
            plot_parameter(
                layers=two_layer_model,
                param_name=param,
                title=title,
                cmap_name=cmap,
                unit_str=unit,
                output_file=os.path.join(output_dir, f"two_layer_{param}_model.png")
            )
            print(f"Created {param} model plot")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
    
    print("All plots generated successfully in the 'velocity_plots' directory")

if __name__ == "__main__":
    main() 