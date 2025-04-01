#!/usr/bin/env python3
"""
Simulate and Visualize SPECFEM3D Data

This script runs a SPECFEM3D simulation and generates visualizations of the results:
1. Shot gather plots
2. Wavefield snapshots

Usage:
    python simulate_and_visualize.py --parameter-set parameter_sets/standard_simulation.json

Features:
- Run a simulation with the specified parameter set
- Generate shot gather plots for velocity, displacement, and acceleration
- Create wavefield snapshots showing wave propagation
- Save all visualizations to the specified output directory
"""

import os
import sys
import argparse
import json
import subprocess
import time
from pathlib import Path
import importlib.util

def load_module(file_path, module_name):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_simulation(specfem_dir, parameter_set=None, nproc=None):
    """
    Run a SPECFEM3D simulation using the specified parameter set.
    
    Args:
        specfem_dir (str): Path to SPECFEM3D installation
        parameter_set (str): Path to parameter set JSON file
        nproc (int): Number of processors to use, overrides parameter set
        
    Returns:
        bool: True if simulation completed successfully
    """
    # Get the path to the simulation script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simulation_script = os.path.join(script_dir, "01_specfem_simulation.py")
    
    if not os.path.exists(simulation_script):
        print(f"Error: Simulation script not found: {simulation_script}")
        return False
    
    # Build command arguments
    cmd = ["python", simulation_script]
    
    # Add parameter set if specified
    if parameter_set:
        param_manager = load_module(os.path.join(script_dir, "param_manager.py"), "param_manager")
        # Check if parameter set exists
        if not os.path.exists(parameter_set):
            print(f"Error: Parameter set not found: {parameter_set}")
            return False
            
        # Load parameter set
        try:
            with open(parameter_set, 'r') as f:
                params = json.load(f)
            print(f"Loaded parameter set: {parameter_set}")
            
            # Apply parameters using param_manager
            param_manager.update_parameters(params)
            print("Applied parameters from set")
            
            # Override nproc if specified
            if nproc is not None:
                param_manager.update_processor_settings(nproc)
                print(f"Overrode processor settings to use {nproc} processors")
        except Exception as e:
            print(f"Error applying parameter set: {e}")
            return False
    
    # Run the simulation
    print(f"Running simulation with: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, cwd=specfem_dir)
        print(f"Simulation started with PID {process.pid}")
        print("Waiting for simulation to complete...")
        
        # Wait for the process to complete with a timeout
        max_wait_time = 3600  # 1 hour maximum
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > max_wait_time:
                print(f"Simulation timed out after {max_wait_time/3600:.1f} hours")
                process.terminate()
                return False
            
            # Print a status update every 5 minutes
            elapsed_minutes = (time.time() - start_time) / 60
            if elapsed_minutes % 5 < 0.1:
                print(f"Simulation running for {elapsed_minutes:.1f} minutes...")
                
            time.sleep(10)
        
        # Check if the process completed successfully
        if process.returncode == 0:
            print(f"Simulation completed successfully in {(time.time() - start_time)/60:.1f} minutes")
            return True
        else:
            print(f"Simulation failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running simulation: {e}")
        return False

def generate_visualizations(specfem_dir, output_dir, data_types=None):
    """
    Generate visualizations of the simulation results.
    
    Args:
        specfem_dir (str): Path to SPECFEM3D installation
        output_dir (str): Path to output directory
        data_types (list): List of data types to visualize (velocity, displacement, acceleration)
        
    Returns:
        bool: True if visualizations were generated successfully
    """
    # Get the path to the visualization scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shot_gather_script = os.path.join(script_dir, "plot_shot_gather.py")
    snapshots_script = os.path.join(script_dir, "plot_wavefield_snapshots.py")
    
    # Check if scripts exist
    if not os.path.exists(shot_gather_script):
        print(f"Error: Shot gather script not found: {shot_gather_script}")
        return False
        
    if not os.path.exists(snapshots_script):
        print(f"Error: Wavefield snapshots script not found: {snapshots_script}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default data types if not specified
    if data_types is None:
        data_types = ["velocity", "displacement"]
    
    success = True
    
    # Generate shot gather plots for each data type
    for data_type in data_types:
        print(f"Generating shot gather plots for {data_type}...")
        cmd = [
            "python", shot_gather_script,
            "--specfem-dir", specfem_dir,
            "--output-dir", output_dir,
            "--data-type", data_type
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Shot gather plots for {data_type} generated successfully")
            else:
                print(f"Error generating shot gather plots for {data_type}: {result.stderr}")
                success = False
        except Exception as e:
            print(f"Error running shot gather script: {e}")
            success = False
    
    # Generate wavefield snapshots for each data type
    for data_type in data_types:
        print(f"Generating wavefield snapshots for {data_type}...")
        output_file = os.path.join(output_dir, f"wavefield_snapshots_{data_type}.png")
        cmd = [
            "python", snapshots_script,
            "--specfem-dir", specfem_dir,
            "--output-file", output_file,
            "--data-type", data_type,
            "--n-snapshots", "9"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Wavefield snapshots for {data_type} generated successfully")
            else:
                print(f"Error generating wavefield snapshots for {data_type}: {result.stderr}")
                success = False
        except Exception as e:
            print(f"Error running wavefield snapshots script: {e}")
            success = False
    
    return success

def main():
    """Main function to parse arguments and run the simulation and visualization."""
    parser = argparse.ArgumentParser(description="Run a SPECFEM3D simulation and generate visualizations")
    
    # Path arguments
    parser.add_argument("--specfem-dir", type=str, default="~/specfem3d",
                      help="Path to SPECFEM3D installation")
    parser.add_argument("--output-dir", type=str, default="./plots",
                      help="Output directory for visualizations")
    
    # Simulation parameters
    parser.add_argument("--parameter-set", type=str,
                      help="Path to parameter set JSON file")
    parser.add_argument("--nproc", type=int,
                      help="Number of processors to use, overrides parameter set")
    parser.add_argument("--skip-simulation", action="store_true",
                      help="Skip running the simulation, just generate visualizations")
    
    # Visualization parameters
    parser.add_argument("--data-types", type=str, nargs="+",
                      default=["velocity", "displacement"],
                      choices=["velocity", "displacement", "acceleration"],
                      help="Data types to visualize")
    
    args = parser.parse_args()
    
    # Expand paths
    specfem_dir = os.path.expanduser(args.specfem_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    # Run simulation if not skipped
    if not args.skip_simulation:
        print("Running SPECFEM3D simulation...")
        success = run_simulation(
            specfem_dir=specfem_dir,
            parameter_set=args.parameter_set,
            nproc=args.nproc
        )
        
        if not success:
            print("Simulation failed, cannot generate visualizations")
            return 1
    
    # Generate visualizations
    print("Generating visualizations...")
    success = generate_visualizations(
        specfem_dir=specfem_dir,
        output_dir=output_dir,
        data_types=args.data_types
    )
    
    if not success:
        print("Some visualizations failed to generate")
        return 1
        
    print(f"All visualizations saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 