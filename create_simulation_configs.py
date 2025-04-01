#!/usr/bin/env python3
"""
Create different simulation configurations for SPECFEM3D.

This script uses the ParamManager class to create different parameter sets for:
- Standard simulation (default)
- High-resolution simulation
- Fast simulation (for testing)
- DAS fiber simulation (with stations in a line for fiber optic sensing)
"""

import os
import sys
from pathlib import Path
from param_manager import ParamManager

# Path setup
specfem_dir = os.path.expanduser("~/specfem3d")
output_dir = os.path.expanduser("~/ml_interpolation/parameter_sets")
Path(output_dir).mkdir(parents=True, exist_ok=True)

def create_standard_simulation():
    """Create parameters for standard simulation."""
    print("Creating standard simulation configuration...")
    pm = ParamManager(specfem_dir)
    
    # Load existing parameters as a starting point
    pm.load_par_file()
    pm.load_mesh_par_file()
    pm.load_source_file()
    
    # Set standard parameters
    pm.update_processor_settings(8)
    
    # Update Par_file parameters
    par_params = {
        'NSTEP': 5000,
        'DT': 0.05,
        'USE_RICKER_TIME_FUNCTION': False,
        'SAVE_SEISMOGRAMS_DISPLACEMENT': True,
        'SAVE_SEISMOGRAMS_VELOCITY': True,
        'SAVE_SEISMOGRAMS_ACCELERATION': True
    }
    pm.set_parameters(par_params, 'par_file')
    
    # Update Mesh_Par_file parameters
    mesh_params = {
        'NEX_XI': 16,
        'NEX_ETA': 16,
        'USE_REGULAR_MESH': True
    }
    pm.set_parameters(mesh_params, 'mesh_par_file')
    
    # Create stations in a regular grid
    pm.create_stations_grid(4, 5, x_min=500, x_max=1500, y_min=500, y_max=1500)
    
    # Save the parameter set
    out_file = os.path.join(output_dir, 'standard_simulation.json')
    pm.save_parameter_set(out_file)
    return out_file

def create_high_resolution_simulation():
    """Create parameters for high-resolution simulation."""
    print("Creating high-resolution simulation configuration...")
    pm = ParamManager(specfem_dir)
    
    # Load existing parameters as a starting point
    pm.load_par_file()
    pm.load_mesh_par_file()
    pm.load_source_file()
    
    # Set high-resolution parameters
    pm.update_processor_settings(16)
    
    # Update Par_file parameters
    par_params = {
        'NSTEP': 10000,
        'DT': 0.025,  # Smaller time step
        'USE_RICKER_TIME_FUNCTION': True,
        'SAVE_SEISMOGRAMS_DISPLACEMENT': True,
        'SAVE_SEISMOGRAMS_VELOCITY': True,
        'SAVE_SEISMOGRAMS_ACCELERATION': True,
        'NTSTEP_BETWEEN_OUTPUT_INFO': 500,
        'NTSTEP_BETWEEN_OUTPUT_SAMPLE': 1,
        'NTSTEP_BETWEEN_FRAMES': 500
    }
    pm.set_parameters(par_params, 'par_file')
    
    # Update Mesh_Par_file parameters
    mesh_params = {
        'NEX_XI': 32,
        'NEX_ETA': 32,
        'USE_REGULAR_MESH': True
    }
    pm.set_parameters(mesh_params, 'mesh_par_file')
    
    # Create more stations in a denser grid
    pm.create_stations_grid(8, 8, x_min=500, x_max=1500, y_min=500, y_max=1500)
    
    # Save the parameter set
    out_file = os.path.join(output_dir, 'high_res_simulation.json')
    pm.save_parameter_set(out_file)
    return out_file

def create_fast_simulation():
    """Create parameters for fast simulation (for testing)."""
    print("Creating fast simulation configuration...")
    pm = ParamManager(specfem_dir)
    
    # Load existing parameters as a starting point
    pm.load_par_file()
    pm.load_mesh_par_file()
    pm.load_source_file()
    
    # Set fast simulation parameters (for testing)
    pm.update_processor_settings(4)
    
    # Update Par_file parameters
    par_params = {
        'NSTEP': 1000,  # Fewer time steps
        'DT': 0.05,
        'USE_RICKER_TIME_FUNCTION': True,
        'SAVE_SEISMOGRAMS_DISPLACEMENT': True,
        'SAVE_SEISMOGRAMS_VELOCITY': True,
        'SAVE_SEISMOGRAMS_ACCELERATION': False,  # Save less data
        'NTSTEP_BETWEEN_OUTPUT_INFO': 100,
        'NTSTEP_BETWEEN_OUTPUT_SAMPLE': 5,  # Sample less frequently
        'NTSTEP_BETWEEN_FRAMES': 100
    }
    pm.set_parameters(par_params, 'par_file')
    
    # Update Mesh_Par_file parameters
    mesh_params = {
        'NEX_XI': 8,  # Coarser mesh
        'NEX_ETA': 8,
        'USE_REGULAR_MESH': True
    }
    pm.set_parameters(mesh_params, 'mesh_par_file')
    
    # Create fewer stations
    pm.create_stations_grid(2, 3, x_min=500, x_max=1500, y_min=500, y_max=1500)
    
    # Save the parameter set
    out_file = os.path.join(output_dir, 'fast_simulation.json')
    pm.save_parameter_set(out_file)
    return out_file

def create_das_fiber_simulation():
    """Create parameters for DAS fiber simulation."""
    print("Creating DAS fiber simulation configuration...")
    pm = ParamManager(specfem_dir)
    
    # Load existing parameters as a starting point
    pm.load_par_file()
    pm.load_mesh_par_file()
    pm.load_source_file()
    
    # Set DAS fiber simulation parameters
    pm.update_processor_settings(8)
    
    # Update Par_file parameters
    par_params = {
        'NSTEP': 5000,
        'DT': 0.05,
        'USE_RICKER_TIME_FUNCTION': True,
        'SAVE_SEISMOGRAMS_DISPLACEMENT': False,
        'SAVE_SEISMOGRAMS_VELOCITY': True,  # Only velocity for DAS
        'SAVE_SEISMOGRAMS_ACCELERATION': False,
        'NTSTEP_BETWEEN_OUTPUT_INFO': 200,
        'NTSTEP_BETWEEN_OUTPUT_SAMPLE': 1,
        'NTSTEP_BETWEEN_FRAMES': 200
    }
    pm.set_parameters(par_params, 'par_file')
    
    # Update Mesh_Par_file parameters
    mesh_params = {
        'NEX_XI': 24,
        'NEX_ETA': 24,
        'USE_REGULAR_MESH': True
    }
    pm.set_parameters(mesh_params, 'mesh_par_file')
    
    # Create stations in a straight line (representing a fiber)
    pm.create_stations_line(50, start_x=500, end_x=1500, y=1000)
    
    # Save the parameter set
    out_file = os.path.join(output_dir, 'das_fiber_simulation.json')
    pm.save_parameter_set(out_file)
    return out_file

def main():
    """Main function to create all simulation configurations."""
    # Create all configurations
    standard_file = create_standard_simulation()
    high_res_file = create_high_resolution_simulation()
    fast_file = create_fast_simulation()
    das_file = create_das_fiber_simulation()
    
    print("\nCreated simulation configurations:")
    print(f"1. Standard simulation: {standard_file}")
    print(f"2. High-resolution simulation: {high_res_file}")
    print(f"3. Fast simulation (for testing): {fast_file}")
    print(f"4. DAS fiber simulation: {das_file}")
    
    print("\nTo use a configuration, run:")
    print("python param_manager.py --load-set <json_file> --save")
    
    # Example: Create directory for a specific simulation
    example_dir = os.path.expanduser("~/specfem3d_simulations/high_res")
    print(f"\nExample - setting up high-resolution simulation:")
    print(f"mkdir -p {example_dir}")
    print(f"python param_manager.py --load-set {high_res_file} --output-dir {example_dir} --save")

if __name__ == "__main__":
    main() 