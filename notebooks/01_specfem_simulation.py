#!/usr/bin/env python3
"""
SPECFEM3D Simulation for Seismic Data Generation - Fixed version

This script demonstrates how to set up and run SPECFEM3D simulations 
to generate synthetic seismic data for the interpolation project.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil

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

# Define simulation parameters
simulation_params = {
    # General simulation parameters
    "NUMBER_OF_SIMULTANEOUS_RUNS": 1,  # Usually set to 1 for single simulations
    "NPROC": 4,  # Number of MPI processes
    "NPROC_XI": 2,  # Number of processes along X direction
    "NPROC_ETA": 2,  # Number of processes along Y direction
    "SIMULATION_TYPE": 1,  # 1 = forward simulation
    "NSTEP": 4000,  # Number of time steps
    "DT": 0.001,  # Time step in seconds
    "MODEL": "default",  # Model type
    "SAVE_FORWARD": ".false.",  # Don't save forward wavefield
    "USE_OLSEN_ATTENUATION": ".false.",  # No attenuation
    "NGNOD": 8,  # Number of nodes per element
    "ABSORBING_CONDITIONS": ".true.",  # Absorbing boundary conditions
    "STACEY_ABSORBING_CONDITIONS": ".true.",  # Use Stacey absorbing boundary conditions
    "ATTENUATION": ".false.",  # No attenuation
    "USE_RICKER_TIME_FUNCTION": ".true.",  # Use Ricker wavelet
    
    # Output parameters
    "SAVE_SEISMOGRAMS_DISPLACEMENT": ".true.",  # Output displacement
    "NTSTEP_BETWEEN_OUTPUT_SEISMOS": 10,  # Output seismograms every 10 steps
    "USE_BINARY_FOR_SEISMOGRAMS": ".false.",  # Save ASCII seismograms
    "SAVE_BINARY_SEISMOGRAMS_SINGLE": ".true.",  # Save binary seismograms
    "SAVE_BINARY_SEISMOGRAMS_DOUBLE": ".false.",  # Don't save double precision binary
    "USE_EXISTING_STATIONS": ".true.",  # Use the STATIONS file we created
    
    # Cartesian mesh parameters
    "LATITUDE_MIN": 0.0,  # Min X coordinate of mesh
    "LATITUDE_MAX": 2000.0,  # Max X coordinate of mesh
    "LONGITUDE_MIN": 0.0,  # Min Y coordinate of mesh
    "LONGITUDE_MAX": 2000.0,  # Max Y coordinate of mesh
    "DEPTH_MIN": 0.0,  # Min Z coordinate of mesh
    "DEPTH_MAX": 2000.0,  # Max Z coordinate of mesh
    "NEX_XI": 40,  # Number of elements along X direction
    "NEX_ETA": 40,  # Number of elements along Y direction
    "NEX_ZETA": 40,  # Number of elements along Z direction
    
    # Additional parameters needed by SPECFEM3D
    "BROADCAST_SAME_MESH_AND_MODEL": ".true.",
    "USE_REGULAR_MESH": ".true.",
    "STACEY_INSTEAD_OF_FREE_SURFACE": ".false.",
    "ROTATE_PML_ACTIVATE": ".false.",
    "PRINT_SOURCE_TIME_FUNCTION": ".false.",
    "GPU_MODE": ".false.",
    "SAVE_MESH_FILES": ".false.",
    "SUPPRESS_UTM_PROJECTION": ".true.",  # Use Cartesian coordinates directly
    "NOISE_TOMOGRAPHY": 0,
    "USE_LDDRK": ".false.",
    "APPROXIMATE_OCEAN_LOAD": ".false.",
    "TOPOGRAPHY": ".false.",
    "ANISOTROPY": ".false.",
    "GRAVITY": ".false.",
    "PML_CONDITIONS": ".false.",
    "PML_INSTEAD_OF_FREE_SURFACE": ".false.",
    "USE_FORCE_POINT_SOURCE": ".false.",
    "USE_SOURCES_RECEIVERS_Z": ".true.",
    "SAVE_SEISMOGRAMS_VELOCITY": ".false.",
    "SAVE_SEISMOGRAMS_ACCELERATION": ".false.",
    "SAVE_SEISMOGRAMS_PRESSURE": ".false.",
    "NTSTEP_BETWEEN_READ_ADJSRC": 0,
    "BINARY_FILE_OUTPUT": ".true.",
    "SU_FORMAT": ".false.",
    "INVERSE_FWI_FULL_PROBLEM": ".false.",
    "INVERSE_STACEY_BOUNDARY_CONDITIONS": ".false."
}

# Define source parameters
source_params = {
    "source_surf": 0,  # Source is inside the medium
    "xs": 1000.0,  # X position in meters
    "ys": 1000.0,  # Y position in meters
    "zs": 500.0,  # Z position in meters (depth positive downward)
    "source_type": 1,  # 1 = force, 2 = moment tensor
    "time_function_type": 2,  # Ricker wavelet
    "name_of_source_file": "",  # Not used for simple source
    "burst_band_width": 0.0,  # Not used for Ricker wavelet
    "f0": 10.0,  # Central frequency in Hz
    "tshift": 0.0,  # Time shift
    "anglesource": 0.0,  # If source_type = 1, angle of force source
    "Mxx": 1.0,  # If source_type = 2, moment tensor components
    "Mxy": 0.0,
    "Mxz": 0.0,
    "Myy": 1.0,
    "Myz": 0.0,
    "Mzz": 1.0
}

# Create station locations
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
stations_file = os.path.join(data_dir, 'STATIONS')
stations.to_csv(stations_file, sep=' ', index=False, header=False)
print(f"Created {n_stations} stations in {stations_file}")

# Initialize simulation with explicit number of processes
simulation = SpecfemSimulation(
    specfem_dir=specfem_dir,
    output_dir=data_dir,
    nproc=simulation_params["NPROC"]
)

# Use the default SPECFEM3D Par_file as template
par_file_template = os.path.join(data_dir, "Par_file.template")

# Also copy the Mesh_Par_file to ensure it's available
mesh_par_source = os.path.join(specfem_dir, "DATA", "meshfem3D_files", "Mesh_Par_file")
mesh_par_dest = os.path.join(specfem_dir, "DATA", "meshfem3D_files", "Mesh_Par_file.bak")
os.makedirs(os.path.dirname(mesh_par_dest), exist_ok=True)
shutil.copy(mesh_par_source, mesh_par_dest)
print(f"Backed up Mesh_Par_file to {mesh_par_dest}")

# Copy CMTSOLUTION file from the example
cmt_source = os.path.join(specfem_dir, "EXAMPLES/applications/homogeneous_halfspace_HEX8_elastic_no_absorbing/DATA/CMTSOLUTION")
cmt_dest = os.path.join(specfem_dir, "DATA/CMTSOLUTION")
if os.path.exists(cmt_source):
    shutil.copy(cmt_source, cmt_dest)
    print(f"Copied CMTSOLUTION file to {cmt_dest}")
else:
    print("Warning: Example CMTSOLUTION file not found")

# Ensure we have a minimal set of interface files
mesh_dir = os.path.join(specfem_dir, "DATA", "meshfem3D_files")
os.makedirs(mesh_dir, exist_ok=True)

# Create no_cavity.dat
with open(os.path.join(mesh_dir, "no_cavity.dat"), "w") as f:
    f.write("0\n")  # No cavities

# Create interfaces.dat for a single-layered medium
with open(os.path.join(mesh_dir, "interfaces.dat"), "w") as f:
    f.write("# number of interfaces\n")
    f.write("1\n")  # At least one interface for topography
    f.write("# interface 1 (topography)\n")
    f.write(".true.\n")  # flat interface
    f.write("interface1.dat\n")
    f.write("# number of spectral elements per layer\n")
    f.write(f"{simulation_params['NEX_ZETA']}")  # All elements in a single layer

# Create a simple flat interface file (for topography)
with open(os.path.join(mesh_dir, "interface1.dat"), "w") as f:
    # Create a 2x2 grid of interface points (all at z=0)
    f.write("2 2\n")  # nx, ny points
    # 4 corner points for a flat interface at z=0
    for ix in range(2):
        for iy in range(2):
            f.write("0.0\n")  # z-coordinate (flat interface)

# Run the full simulation with our complete parameter set
success = simulation.run_full_simulation(
    par_file_template=par_file_template,
    simulation_params=simulation_params,
    source_params=source_params,
    stations_list=stations.to_dict('records')
)

# Verify that the Par_file was properly created and copied
print("\nPar_file verification:")
par_file_output = os.path.join(data_dir, "Par_file")
specfem_par_file = os.path.join(specfem_dir, "DATA", "Par_file")
print(f"1. Output Par_file exists: {os.path.exists(par_file_output)}")
print(f"2. SPECFEM DATA Par_file exists: {os.path.exists(specfem_par_file)}")

# Compare first few lines to verify content
if os.path.exists(par_file_output) and os.path.exists(specfem_par_file):
    with open(par_file_output, 'r') as f:
        output_lines = f.readlines()[:5]
    with open(specfem_par_file, 'r') as f:
        specfem_lines = f.readlines()[:5]
    
    print("Output Par_file first 5 lines:")
    print(''.join(output_lines))
    print("SPECFEM Par_file first 5 lines:")
    print(''.join(specfem_lines))

# Check if the Par_file has the NUMBER_OF_SIMULTANEOUS_RUNS parameter
if os.path.exists(specfem_par_file):
    with open(specfem_par_file, 'r') as f:
        content = f.read()
        has_num_runs = 'NUMBER_OF_SIMULTANEOUS_RUNS' in content
        print(f"3. SPECFEM Par_file has NUMBER_OF_SIMULTANEOUS_RUNS: {has_num_runs}")

if success:
    print("Simulation completed successfully!")
    
    # Function to read SPECFEM3D seismogram files
    def read_specfem_seismogram(file_path):
        """Read a SPECFEM3D seismogram file and return time and amplitude."""
        try:
            data = np.loadtxt(file_path, skiprows=0)
            times = data[:, 0]
            amplitude = data[:, 1]
            return times, amplitude
        except Exception as e:
            print(f"Error reading seismogram file {file_path}: {e}")
            return np.array([]), np.array([])
    
    # Plot individual traces
    seismogram_dir = os.path.join(data_dir, 'OUTPUT_FILES')
    
    # Find seismogram files for the first station
    station_code = "ST001"
    seismogram_files = []
    for root, dirs, files in os.walk(seismogram_dir):
        for file in files:
            if station_code in file and file.endswith(".semd"):
                seismogram_files.append(os.path.join(root, file))
    
    if seismogram_files:
        # Read the first component (e.g., BXZ for vertical)
        times, amplitude = read_specfem_seismogram(seismogram_files[0])
        if len(times) > 0:
            # Now use the correct function signature
            plot_seismic_trace(times, amplitude, title=f'Seismic Trace - {station_code}')
        else:
            print(f"No data found for station {station_code}")
    else:
        print(f"No seismogram files found for station {station_code}")
    
    # For plot_seismic_gather, you need to read all station data first
    # This is simplified and would need to be expanded based on your actual data structure
    try:
        # Read all stations for one component (e.g., Z)
        all_traces = []
        for i in range(1, n_stations + 1):
            station = f"ST{i:03d}"
            # Find Z component file (usually BXZ)
            component_file = None
            for root, dirs, files in os.walk(seismogram_dir):
                for file in files:
                    if station in file and "BXZ" in file and file.endswith(".semd"):
                        component_file = os.path.join(root, file)
                        break
            
            if component_file and os.path.exists(component_file):
                _, trace = read_specfem_seismogram(component_file)
                if len(trace) > 0:
                    all_traces.append(trace)
        
        if all_traces:
            # Convert to numpy array
            data = np.array(all_traces)
            plot_seismic_gather(data)
        else:
            print("No seismic traces found to create gather")
    except Exception as e:
        print(f"Error creating seismic gather: {e}")
else:
    print("Simulation failed. Check logs for details.")