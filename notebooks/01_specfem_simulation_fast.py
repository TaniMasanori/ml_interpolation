#!/usr/bin/env python3
"""
SPECFEM3D Simulation for Seismic Data Generation - Fast version

This script runs a SPECFEM3D simulation using a pre-built example mesh.
It skips the visualization steps for faster execution.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil
import subprocess

# Add the project root to path for imports
sys.path.append('..')

# Import project modules
from src.simulation.specfem_runner import SpecfemSimulation
from src.utils.logging_utils import setup_logging
from src.utils.plot_utils import plot_seismic_trace, plot_seismic_gather

# Set up logging
logger = setup_logging(level='INFO')

# Paths
specfem_dir = os.path.expanduser("~/specfem3d")  # Path to SPECFEM3D installation
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
data_dir = os.path.join(project_root, "data/synthetic/raw/simulation1")
templates_dir = os.path.join(project_root, "specfem_simulations")

# Ensure directories exist
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(templates_dir).mkdir(parents=True, exist_ok=True)

# Copy files from a working example
example_dir = os.path.join(specfem_dir, "EXAMPLES", "applications", "homogeneous_halfspace_HEX8_elastic_no_absorbing")
example_data_dir = os.path.join(example_dir, "DATA")
dest_data_dir = os.path.join(specfem_dir, "DATA")

# Define the files we need to copy
files_to_copy = [
    "Par_file",
    "CMTSOLUTION",
    "STATIONS"
]

# Copy essential files from the example to the main DATA directory
for file in files_to_copy:
    source_file = os.path.join(example_data_dir, file)
    dest_file = os.path.join(dest_data_dir, file)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)
        print(f"Copied {file} from example to {dest_file}")
    else:
        print(f"Warning: {file} not found in example directory")

# Copy the MESH-default directory
example_mesh_dir = os.path.join(example_dir, "MESH-default")
dest_mesh_dir = os.path.join(specfem_dir, "DATA", "MESH-default")
if os.path.exists(example_mesh_dir):
    # If destination dir exists, clear it first
    if os.path.exists(dest_mesh_dir):
        shutil.rmtree(dest_mesh_dir)
    # Copy the directory
    shutil.copytree(example_mesh_dir, dest_mesh_dir)
    print(f"Copied MESH-default directory from example")
else:
    print("Warning: MESH-default directory not found in example. Please make sure it exists.")

# Create our own STATIONS file with multiple stations
n_stations = 20
stations = pd.DataFrame({
    'name': [f'ST{i:03d}' for i in range(1, n_stations + 1)],
    'network': ['GE'] * n_stations,
    'lat': np.linspace(500, 1500, n_stations),  # X coordinates
    'lon': [1000.0] * n_stations,  # Y coordinates
    'elevation': [0.0] * n_stations,
    'burial': [0.0] * n_stations
})

# Save stations to file
stations_file = os.path.join(dest_data_dir, 'STATIONS')
stations.to_csv(stations_file, sep=' ', index=False, header=False)
print(f"Created {n_stations} stations in {stations_file}")

# Ensure output directories exist and are clean
for dir_name in ["OUTPUT_FILES"]:
    dir_path = os.path.join(specfem_dir, dir_name)
    if os.path.exists(dir_path):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Cleaned {dir_name} directory")
    else:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created {dir_name} directory")

# Create DATABASES_MPI directory
database_dir = os.path.join(specfem_dir, "OUTPUT_FILES", "DATABASES_MPI")
if os.path.exists(database_dir):
    for item in os.listdir(database_dir):
        item_path = os.path.join(database_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print(f"Cleaned DATABASES_MPI directory")
else:
    os.makedirs(database_dir, exist_ok=True)
    print(f"Created DATABASES_MPI directory")

# Number of processors to use
nproc = 4

# Now follow the steps in HOWTO_run_this_example.txt
print("\nRunning simulation as per example instructions:")

# Step 1: Decompose the mesh
print("Step 1: Decomposing the mesh...")
cmd = f"cd {specfem_dir} && ./bin/xdecompose_mesh {nproc} DATA/MESH-default/ OUTPUT_FILES/DATABASES_MPI/"
print(f"Running command: {cmd}")
try:
    subprocess.run(cmd, shell=True, check=True)
    print("Mesh decomposed successfully.")
    mesh_success = True
except subprocess.CalledProcessError:
    print("Mesh decomposition failed.")
    mesh_success = False

if mesh_success:
    # Step 2: Generate databases
    print("Step 2: Generating databases...")
    cmd = f"cd {specfem_dir} && mpirun -np {nproc} ./bin/xgenerate_databases"
    print(f"Running command: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("Databases generated successfully.")
        database_success = True
    except subprocess.CalledProcessError:
        print("Database generation failed.")
        database_success = False

    if database_success:
        # Step 3: Run the simulation
        print("Step 3: Running the simulation...")
        cmd = f"cd {specfem_dir} && mpirun -np {nproc} ./bin/xspecfem3D"
        print(f"Running command: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            print("Simulation completed successfully!")
            simulation_success = True
        except subprocess.CalledProcessError:
            print("Simulation failed.")
            simulation_success = False
    else:
        simulation_success = False
else:
    print("Simulation cannot proceed due to mesh decomposition failure.")
    simulation_success = False

# Verify that the Par_file was properly copied
print("\nPar_file verification:")
specfem_par_file = os.path.join(specfem_dir, "DATA", "Par_file")
print(f"SPECFEM DATA Par_file exists: {os.path.exists(specfem_par_file)}")

if simulation_success:
    # Count the number of seismogram files
    seismogram_dir = os.path.join(specfem_dir, 'OUTPUT_FILES')
    seismogram_count = 0
    for root, dirs, files in os.walk(seismogram_dir):
        for file in files:
            if file.endswith(".semd"):
                seismogram_count += 1
    
    print(f"\nSimulation Statistics:")
    print(f"Number of stations: {n_stations}")
    print(f"Number of seismogram files generated: {seismogram_count}")
    print(f"Expected files per station: 3 components x 3 file types = 9")
    print(f"Total expected files: {n_stations * 9}")
    print(f"Output directory: {seismogram_dir}")
    print("\nThe simulation has completed successfully!")
else:
    print("Simulation failed. Check logs for details.")