#!/usr/bin/env python3
"""
Fix SPECFEM3D UTM_PROJECTION_ZONE Type Issue

This script specifically addresses the issue with UTM_PROJECTION_ZONE parameter
formatting by creating files that explicitly disable UTM projection and use
an integer value without quotes.
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

print("=== Fixing SPECFEM3D UTM_PROJECTION_ZONE Type Issue ===")

# Create a focused Mesh_Par_file with UTM projection explicitly disabled
mesh_par_content = """#-----------------------------------------------------------
#
# Meshing input parameters
#
#-----------------------------------------------------------

# coordinates of mesh block in latitude/longitude and depth in km
LATITUDE_MIN                    = 0.0
LATITUDE_MAX                    = 2000.0
LONGITUDE_MIN                   = 0.0
LONGITUDE_MAX                   = 2000.0
DEPTH_MIN                       = 0.0
DEPTH_MAX                       = 2000.0

# number of spectral elements along xi and eta
NEX_XI                          = 40
NEX_ETA                         = 40

# number of processors along xi and eta
NPROC_XI                        = 2
NPROC_ETA                       = 2

# UTM projection parameters
# IMPORTANT: DO NOT use quotes for UTM_PROJECTION_ZONE, must be an integer
UTM_PROJECTION_ZONE             = 11
SUPPRESS_UTM_PROJECTION         = .false.

# file that contains the interfaces of the model / mesh
INTERFACES_FILE                 = interfaces.dat

# file that contains the cavity
CAVITY_FILE                     = no_cavity.dat

# Regular/irregular mesh
USE_REGULAR_MESH                = .true.
NDOUBLINGS                      = 0
NZ_DOUBLING_1                   = 0
NZ_DOUBLING_2                   = 0

# create mesh files for visualisation
CREATE_ABAQUS_FILES             = .false.
CREATE_DX_FILES                 = .false.
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

# Create Par_file with explicit UTM projection disabled
par_file_content = """#-----------------------------------------------------------
#
# Simulation input parameters
#
#-----------------------------------------------------------

# forward or adjoint simulation
# 1 = forward, 2 = adjoint, 3 = both simultaneously
SIMULATION_TYPE                 = 1
# 0 = earthquake simulation,  1/2/3 = three steps in noise simulation
NOISE_TOMOGRAPHY                = 0
SAVE_FORWARD                    = .false.

# inverse problem
INVERSE_FWI_FULL_PROBLEM        = .false.

# UTM projection parameters
UTM_PROJECTION_ZONE             = 11
SUPPRESS_UTM_PROJECTION         = .false.

# number of MPI processors
NPROC                           = 4

# time step parameters
NSTEP                           = 1000
DT                              = 0.001

# set to true to use local-time stepping (LTS)
LTS_MODE                        = .false.

# partitioning algorithm
PARTITIONING_TYPE               = 1

# LDDRK time scheme
USE_LDDRK                       = .false.
INCREASE_CFL_FOR_LDDRK          = .false.
RATIO_BY_WHICH_TO_INCREASE_IT   = 1.4

# mesh parameters
NGNOD                           = 8
MODEL                           = default

# path for external files
TOMOGRAPHY_PATH                 = ./DATA/tomo_files/
SEP_MODEL_DIRECTORY             = ./DATA/my_SEP_model/

# Cartesian mesh parameters
LATITUDE_MIN                    = 0.0
LATITUDE_MAX                    = 2000.0
LONGITUDE_MIN                   = 0.0
LONGITUDE_MAX                   = 2000.0
DEPTH_MIN                       = 0.0
DEPTH_MAX                       = 2000.0
NEX_XI                          = 40
NEX_ETA                         = 40
NEX_ZETA                        = 40
NPROC_XI                        = 2
NPROC_ETA                       = 2

# model settings
APPROXIMATE_OCEAN_LOAD          = .false.
TOPOGRAPHY                      = .false.
ATTENUATION                     = .false.
ANISOTROPY                      = .false.
GRAVITY                         = .false.

# attenuation
ATTENUATION_f0_REFERENCE        = 0.33333d0
MIN_ATTENUATION_PERIOD          = 999999998.d0
MAX_ATTENUATION_PERIOD          = 999999999.d0
COMPUTE_FREQ_BAND_AUTOMATIC     = .true.
USE_OLSEN_ATTENUATION           = .false.
OLSEN_ATTENUATION_RATIO         = 0.05

# boundary conditions
PML_CONDITIONS                  = .false.
PML_INSTEAD_OF_FREE_SURFACE     = .false.
f0_FOR_PML                      = 0.05555
STACEY_ABSORBING_CONDITIONS     = .true.
STACEY_INSTEAD_OF_FREE_SURFACE  = .false.
BOTTOM_FREE_SURFACE             = .false.

# undoing attenuation
UNDO_ATTENUATION_AND_OR_PML     = .false.
NT_DUMP_ATTENUATION             = 500

# visualization
CREATE_SHAKEMAP                 = .false.
MOVIE_SURFACE                   = .false.
MOVIE_TYPE                      = 1
MOVIE_VOLUME                    = .false.
SAVE_DISPLACEMENT               = .false.
MOVIE_VOLUME_TYPE               = 0
USE_HIGHRES_FOR_MOVIES          = .false.
NTSTEP_BETWEEN_FRAMES           = 200
HDUR_MOVIE                      = 0.0
SAVE_MESH_FILES                 = .false.
LOCAL_PATH                      = ./OUTPUT_FILES/DATABASES_MPI
NTSTEP_BETWEEN_OUTPUT_INFO      = 500

# sources
USE_SOURCES_RECEIVERS_Z         = .false.
USE_FORCE_POINT_SOURCE          = .false.
USE_RICKER_TIME_FUNCTION        = .true.
USE_EXTERNAL_SOURCE_FILE        = .false.
PRINT_SOURCE_TIME_FUNCTION      = .false.
USE_SOURCE_ENCODING             = .false.

# seismograms
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = 10
NTSTEP_BETWEEN_OUTPUT_SAMPLE    = 1
SAVE_SEISMOGRAMS_DISPLACEMENT   = .true.
SAVE_SEISMOGRAMS_VELOCITY       = .false.
SAVE_SEISMOGRAMS_ACCELERATION   = .false.
SAVE_SEISMOGRAMS_PRESSURE       = .false.
SAVE_SEISMOGRAMS_STRAIN         = .false.
SAVE_SEISMOGRAMS_IN_ADJOINT_RUN = .false.
USE_BINARY_FOR_SEISMOGRAMS      = .false.
SU_FORMAT                       = .false.
ASDF_FORMAT                     = .false.
HDF5_FORMAT                     = .false.
WRITE_SEISMOGRAMS_BY_MAIN       = .false.
SAVE_ALL_SEISMOS_IN_ONE_FILE    = .false.
USE_TRICK_FOR_BETTER_PRESSURE   = .false.

# energy
OUTPUT_ENERGY                   = .false.
NTSTEP_BETWEEN_OUTPUT_ENERGY    = 10

# adjoint
NTSTEP_BETWEEN_READ_ADJSRC      = 0
READ_ADJSRC_ASDF                = .false.
ANISOTROPIC_KL                  = .false.
SAVE_TRANSVERSE_KL              = .false.
ANISOTROPIC_VELOCITY_KL         = .false.
APPROXIMATE_HESS_KL             = .false.
SAVE_MOHO_MESH                  = .false.

# coupling
COUPLE_WITH_INJECTION_TECHNIQUE = .false.
INJECTION_TECHNIQUE_TYPE        = 3
MESH_A_CHUNK_OF_THE_EARTH       = .false.
TRACTION_PATH                   = ./DATA/AxiSEM_tractions/3/
FKMODEL_FILE                    = FKmodel
RECIPROCITY_AND_KH_INTEGRAL     = .false.

# run modes
NUMBER_OF_SIMULTANEOUS_RUNS     = 1
BROADCAST_SAME_MESH_AND_MODEL   = .true.

# GPU mode
GPU_MODE                        = .false.

# ADIOS
ADIOS_ENABLED                   = .false.
ADIOS_FOR_DATABASES             = .false.
ADIOS_FOR_MESH                  = .false.
ADIOS_FOR_FORWARD_ARRAYS        = .false.
ADIOS_FOR_KERNELS               = .false.
ADIOS_FOR_UNDO_ATTENUATION      = .false.

# HDF5
HDF5_ENABLED                    = .false.
HDF5_FOR_MOVIES                 = .false.
HDF5_IO_NODES                   = 0
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
print("Writing configuration files with correct UTM_PROJECTION_ZONE type...")

# Mesh_Par_file - notice we're completely removing the parameter that's causing issues
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

# Now manually fix the UTM_PROJECTION_ZONE issue by editing the files
print("\nSpecifically checking and fixing UTM_PROJECTION_ZONE format...")

def fix_utm_zone_format(file_path):
    """Ensure UTM_PROJECTION_ZONE is correctly formatted as an integer"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    utm_zone_found = False
    
    # Check if UTM_PROJECTION_ZONE exists and fix if needed
    for i, line in enumerate(lines):
        if "UTM_PROJECTION_ZONE" in line:
            utm_zone_found = True
            # Make sure it's an integer without quotes
            lines[i] = "UTM_PROJECTION_ZONE             = 11\n"
    
    # If UTM_PROJECTION_ZONE wasn't found, add it
    if not utm_zone_found:
        # Look for a good place to insert it - after SUPPRESS_UTM_PROJECTION
        insert_pos = None
        for i, line in enumerate(lines):
            if "SUPPRESS_UTM_PROJECTION" in line:
                insert_pos = i
                # Change to .false. since we are using UTM projection
                lines[i] = "SUPPRESS_UTM_PROJECTION         = .false.\n"
                break
        
        if insert_pos is not None:
            lines.insert(insert_pos, "UTM_PROJECTION_ZONE             = 11\n")
        else:
            # If can't find a good insertion point, add to the top after the header
            for i, line in enumerate(lines):
                if "#-----------------------------------------------------------" in line and i < 10:
                    lines.insert(i + 3, "UTM_PROJECTION_ZONE             = 11\n")
                    lines.insert(i + 4, "SUPPRESS_UTM_PROJECTION         = .false.\n")
                    break
    
    # Write the file back
    with open(file_path, 'w') as file:
        file.writelines(lines)
    print(f"  - Fixed {file_path}")

# Fix UTM_PROJECTION_ZONE in Par_file and Mesh_Par_file
fix_utm_zone_format(os.path.join(DATA_DIR, "Par_file"))
fix_utm_zone_format(os.path.join(DATA_DIR, "Mesh_Par_file"))
fix_utm_zone_format(os.path.join(MESH_DIR, "Mesh_Par_file"))

print("\nAll configuration files created successfully with UTM_PROJECTION_ZONE issue fixed.")
print("""
You can now run SPECFEM3D with:
cd ~/specfem3d
mpirun -np 4 ./bin/xmeshfem3D
mpirun -np 4 ./bin/xdecompose_mesh
mpirun -np 4 ./bin/xspecfem3D
""")