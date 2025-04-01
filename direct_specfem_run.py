#!/usr/bin/env python3
"""
Direct SPECFEM3D Simulation Run

This script creates minimal, direct files for SPECFEM3D and runs the simulation,
bypassing the complex parameter setup that might cause issues.
"""
import os
import shutil
import subprocess
import sys

# Define paths
SPECFEM_DIR = os.path.expanduser("~/specfem3d")
DATA_DIR = os.path.join(SPECFEM_DIR, "DATA")
MESH_DIR = os.path.join(DATA_DIR, "meshfem3D_files")
OUTPUT_DIR = os.path.join(SPECFEM_DIR, "OUTPUT_FILES")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_DIR = os.path.join(PROJECT_DIR, "data/synthetic/raw/simulation1")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MESH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "DATABASES_MPI"), exist_ok=True)
os.makedirs(SIMULATION_DIR, exist_ok=True)

print("=== Creating minimal SPECFEM3D configuration ===")

# Create minimal Mesh_Par_file without UTM projection
mesh_par_content = """
# coordinates of mesh block in latitude/longitude and depth in km
LATITUDE_MIN                    = 0.0
LATITUDE_MAX                    = 2000.0
LONGITUDE_MIN                   = 0.0
LONGITUDE_MAX                   = 2000.0
DEPTH_MIN                       = 0.0
DEPTH_MAX                       = 2000.0

# UTM projection parameters - EXPLICITLY DISABLED
SUPPRESS_UTM_PROJECTION         = .true.

# file that contains the interfaces of the model / mesh
INTERFACES_FILE                 = interfaces.dat

# file that contains the cavity
CAVITY_FILE                     = no_cavity.dat

# number of elements at the surface along edges of the mesh at the surface
NEX_XI                          = 40
NEX_ETA                         = 40

# number of MPI processors along xi and eta
NPROC_XI                        = 2
NPROC_ETA                       = 2

# Regular/irregular mesh
USE_REGULAR_MESH                = .true.
NDOUBLINGS                      = 0
NZ_DOUBLING_1                   = 0

# create mesh files for visualisation
CREATE_VTK_FILES                = .true.

# path to store the databases files
LOCAL_PATH                      = ./DATABASES_MPI

# number of materials
NMATERIALS                      = 1
# define the different materials in the model as:
# #material_id  #rho  #vp  #vs  #Q_Kappa  #Q_mu  #anisotropy_flag  #domain_id
1 2700.0 3500.0 2000.0 9999.0 9999.0 0 2

# number of regions
NREGIONS                        = 1
# define the different regions of the model as :
#NEX_XI_BEGIN  #NEX_XI_END  #NEX_ETA_BEGIN  #NEX_ETA_END  #NZ_BEGIN #NZ_END  #material_id
1 40 1 40 1 40 1
"""

# Create Par_file with minimal parameters, including the missing parameters
par_file_content = """
# simulation type
SIMULATION_TYPE                 = 1
NOISE_TOMOGRAPHY                = 0
SAVE_FORWARD                    = .false.

# UTM projection - EXPLICITLY DISABLED
SUPPRESS_UTM_PROJECTION         = .true.

# number of MPI processors
NPROC                           = 4

# time step parameters
NSTEP                           = 1000
DT                              = 0.001

# model
NGNOD                           = 8
MODEL                           = default

# attenuation
ATTENUATION                     = .false.

# boundaries
STACEY_ABSORBING_CONDITIONS     = .true.
STACEY_INSTEAD_OF_FREE_SURFACE  = .false.

# source
USE_RICKER_TIME_FUNCTION        = .true.

# output
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = 10
SAVE_SEISMOGRAMS_DISPLACEMENT   = .true.
USE_BINARY_FOR_SEISMOGRAMS      = .false.

# run modes
NUMBER_OF_SIMULTANEOUS_RUNS     = 1
BROADCAST_SAME_MESH_AND_MODEL   = .true.

# local path
LOCAL_PATH                      = ./OUTPUT_FILES/DATABASES_MPI
"""

# Create interface file
interfaces_content = """# number of interfaces
1
# for each interface below, we give the number of points in xi and eta directions
2 2
# and then x,y,z for all these points (xi and eta being the first and second directions respectively)
0.0 0.0 0.0
2000.0 0.0 0.0
0.0 2000.0 0.0
2000.0 2000.0 0.0
"""

# Create cavity file
cavity_content = "# No cavity"

# Create CMTSOLUTION file
cmt_content = """PDE 2000 1 1 0 0 0.0 1000.0 1000.0 500.0 4.5 0.0 0.0 sample_event
event name:       sample_event
time shift:       0.0
half duration:    0.5
latitude:         1000.0
longitude:        1000.0
depth:            500.0
Mrr:              1.0
Mtt:              1.0
Mpp:              1.0
Mrt:              0.0
Mrp:              0.0
Mtp:              0.0
"""

# Create STATIONS file
stations_content = ""
for i in range(1, 11):  # Reduced number of stations for simplicity
    x = 500.0 + (1000.0 * (i-1) / 10)
    stations_content += f"ST{i:03d} GE {x} 1000.0 0.0 0.0\n"

# Create SOURCE file
source_content = """source_surf = 0
xs = 1000.0
ys = 1000.0
zs = 500.0
source_type = 1
time_function_type = 2
name_of_source_file = 
burst_band_width = 0.0
f0 = 10.0
tshift = 0.0
anglesource = 0.0
Mxx = 1.0
Mxy = 0.0
Mxz = 0.0
Myy = 1.0
Myz = 0.0
Mzz = 1.0
"""

# Write all files
print("Writing configuration files...")

# Mesh_Par_file
with open(os.path.join(MESH_DIR, "Mesh_Par_file"), 'w') as f:
    f.write(mesh_par_content)
print(f"- Created {os.path.join(MESH_DIR, 'Mesh_Par_file')}")

# Copy to DATA directory as well
with open(os.path.join(DATA_DIR, "Mesh_Par_file"), 'w') as f:
    f.write(mesh_par_content)
print(f"- Created {os.path.join(DATA_DIR, 'Mesh_Par_file')}")

# Par_file
with open(os.path.join(DATA_DIR, "Par_file"), 'w') as f:
    f.write(par_file_content)
print(f"- Created {os.path.join(DATA_DIR, 'Par_file')}")

# interfaces.dat
with open(os.path.join(MESH_DIR, "interfaces.dat"), 'w') as f:
    f.write(interfaces_content)
print(f"- Created {os.path.join(MESH_DIR, 'interfaces.dat')}")

# no_cavity.dat
with open(os.path.join(MESH_DIR, "no_cavity.dat"), 'w') as f:
    f.write(cavity_content)
print(f"- Created {os.path.join(MESH_DIR, 'no_cavity.dat')}")

# CMTSOLUTION
with open(os.path.join(DATA_DIR, "CMTSOLUTION"), 'w') as f:
    f.write(cmt_content)
print(f"- Created {os.path.join(DATA_DIR, 'CMTSOLUTION')}")

# STATIONS
with open(os.path.join(DATA_DIR, "STATIONS"), 'w') as f:
    f.write(stations_content)
print(f"- Created {os.path.join(DATA_DIR, 'STATIONS')}")

# Copy STATIONS to simulation directory
with open(os.path.join(SIMULATION_DIR, "STATIONS"), 'w') as f:
    f.write(stations_content)
print(f"- Created {os.path.join(SIMULATION_DIR, 'STATIONS')}")

# SOURCE
with open(os.path.join(DATA_DIR, "SOURCE"), 'w') as f:
    f.write(source_content)
print(f"- Created {os.path.join(DATA_DIR, 'SOURCE')}")

# Copy SOURCE to simulation directory
with open(os.path.join(SIMULATION_DIR, "SOURCE"), 'w') as f:
    f.write(source_content)
print(f"- Created {os.path.join(SIMULATION_DIR, 'SOURCE')}")

# Create DATABASES_MPI directory in OUTPUT_FILES if it doesn't exist
os.makedirs(os.path.join(OUTPUT_DIR, "DATABASES_MPI"), exist_ok=True)
print(f"- Ensured {os.path.join(OUTPUT_DIR, 'DATABASES_MPI')} directory exists")

print("\nAll configuration files created successfully.")

# Run the simulation
print("\n=== Running SPECFEM3D Simulation ===")

# Function to run a command and capture output
def run_command(cmd, name, cwd=None):
    print(f"\n--- Running {name} ---")
    print(f"Executing: {cmd}")
    
    try:
        # Use subprocess with Popen to capture output in real-time
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=cwd,
            universal_newlines=True  # Text mode
        )
        
        # Read output in real-time
        stdout_lines = []
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            stdout_lines.append(line)
            sys.stdout.flush()
            
        # Wait for process to complete
        process.wait()
        
        # Get the return code
        return_code = process.returncode
        
        if return_code != 0:
            stderr = process.stderr.read()
            print(f"ERROR: {name} failed with return code {return_code}")
            print(f"STDERR: {stderr}")
            return False
        else:
            print(f"{name} completed successfully")
            return True
    except Exception as e:
        print(f"Error running {name}: {e}")
        return False

# Run the simulation steps with absolute paths
success = True

# Step 1: Run the mesher with the working directory set to SPECFEM_DIR
if success:
    cmd = "mpirun -np 4 ./bin/xmeshfem3D"
    success = run_command(cmd, "Mesher", cwd=SPECFEM_DIR)

# Step 2: Run the partitioner
if success:
    cmd = "mpirun -np 4 ./bin/xdecompose_mesh"
    success = run_command(cmd, "Partitioner", cwd=SPECFEM_DIR)

# Step 3: Run the solver
if success:
    cmd = "mpirun -np 4 ./bin/xspecfem3D"
    success = run_command(cmd, "Solver", cwd=SPECFEM_DIR)

# Step 4: Copy output files to simulation directory
if success:
    print("\n=== Copying output files to simulation directory ===")
    try:
        file_count = 0
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(".semd") or filename.endswith(".semp") or filename.endswith(".sema"):
                src_file = os.path.join(OUTPUT_DIR, filename)
                dst_file = os.path.join(SIMULATION_DIR, filename)
                shutil.copy(src_file, dst_file)
                file_count += 1
                print(f"Copied {filename}")
        
        print(f"Copied {file_count} output files to {SIMULATION_DIR}")
    except Exception as e:
        print(f"Error copying output files: {e}")
        success = False

if success:
    print("\n=== Simulation Completed Successfully ===")
    print(f"Output files are available in:")
    print(f"- SPECFEM3D output: {OUTPUT_DIR}")
    print(f"- Project simulation: {SIMULATION_DIR}")
    print("\nNext steps:")
    print("1. Run DAS conversion (notebooks/02_das_conversion.ipynb)")
    print("2. Preprocess the data (notebooks/03_data_preprocessing.ipynb)")
    print("3. Train the model (notebooks/04_model_training.ipynb)")
else:
    print("\n=== Simulation Failed ===")
    print("Please check the error messages above for details.")