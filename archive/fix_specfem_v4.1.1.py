#!/usr/bin/env python3
"""
SPECFEM3D v4.1.1 Complete Parameter Fix
This script creates a fully compliant Par_file and Mesh_Par_file for SPECFEM3D v4.1.1
which requires all parameters to be explicitly defined (even with default values).
"""
import os
import shutil
import re
from pathlib import Path

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

print(f"=== SPECFEM3D v4.1.1 Complete Parameter Fix ===")
print(f"SPECFEM3D directory: {SPECFEM_DIR}")
print(f"Project output directory: {SIMULATION_DIR}")

# Clean directories first
print("\nCleaning directories...")
# Clean OUTPUT_FILES directory
for item in os.listdir(OUTPUT_DIR):
    item_path = os.path.join(OUTPUT_DIR, item)
    if os.path.isfile(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path) and item != "DATABASES_MPI":
        shutil.rmtree(item_path)

# Clean DATA directory
for item in os.listdir(DATA_DIR):
    item_path = os.path.join(DATA_DIR, item)
    if item != "meshfem3D_files" and not item.startswith("."):
        if os.path.isfile(item_path):
            os.remove(item_path)

# Create a complete Par_file for v4.1.1
par_file = os.path.join(DATA_DIR, "Par_file")
print(f"Creating complete Par_file at {par_file}")

with open(par_file, 'w') as f:
    f.write("""#-----------------------------------------------------------
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

# solve a full FWI inverse problem from a single calling program with no I/Os, storing everything in memory,
# or run a classical forward or adjoint problem only and save the seismograms and/or sensitivity kernels to disk (with costlier I/Os)
INVERSE_FWI_FULL_PROBLEM        = .false.

# UTM projection parameters
# Use a negative zone number for the Southern hemisphere:
# The Northern hemisphere corresponds to zones +1 to +60,
# The Southern hemisphere corresponds to zones -1 to -60.
UTM_PROJECTION_ZONE             = 11
SUPPRESS_UTM_PROJECTION         = .true.

# number of MPI processors
NPROC                           = 4

# time step parameters
NSTEP                           = 4000
DT                              = 0.001

# set to true to use local-time stepping (LTS)
LTS_MODE                        = .false.

# Partitioning algorithm for decompose_mesh
# choose partitioner: 1==SCOTCH (default), 2==METIS, 3==PATOH, 4==ROWS_PART
PARTITIONING_TYPE               = 1

#-----------------------------------------------------------
#
# LDDRK time scheme
#
#-----------------------------------------------------------
USE_LDDRK                       = .false.
INCREASE_CFL_FOR_LDDRK          = .false.
RATIO_BY_WHICH_TO_INCREASE_IT   = 1.4

#-----------------------------------------------------------
#
# Mesh
#
#-----------------------------------------------------------

# Number of nodes for 2D and 3D shape functions for hexahedra.
# We use either 8-node mesh elements (bricks) or 27-node elements.
# If you use our internal mesher, the only option is 8-node bricks (27-node elements are not supported).
NGNOD                           = 8

# models:
# available options are:
#   default (model parameters described by mesh properties)
# 1D models available are:
#   1d_prem,1d_socal,1d_cascadia
# 3D models available are:
#   aniso,external,gll,salton_trough,tomo,SEP,coupled,...
MODEL                           = default

# path for external tomographic models files
TOMOGRAPHY_PATH                 = ./DATA/tomo_files/
# if you are using a SEP model (oil-industry format)
SEP_MODEL_DIRECTORY             = ./DATA/my_SEP_model/

#-----------------------------------------------------------
#
# Parameters for the Cartesian mesh creation
#
#-----------------------------------------------------------

# Mesh dimensions for Cartesian grid (triplanar option)
LATITUDE_MIN                    = 0.0
LATITUDE_MAX                    = 2000.0
LONGITUDE_MIN                   = 0.0
LONGITUDE_MAX                   = 2000.0
DEPTH_MIN                       = 0.0
DEPTH_MAX                       = 2000.0

# Number of elements along each edge
NEX_XI                          = 40
NEX_ETA                         = 40
NEX_ZETA                        = 40

# This affects the mesher but is not in Mesh_Par_file
NPROC_XI                        = 2
NPROC_ETA                       = 2

#-----------------------------------------------------------

# parameters describing the model
APPROXIMATE_OCEAN_LOAD          = .false.
TOPOGRAPHY                      = .false.
ATTENUATION                     = .false.
ANISOTROPY                      = .false.
GRAVITY                         = .false.

# in case of attenuation, reference frequency in Hz at which the velocity values in the velocity model are given (unused otherwise)
ATTENUATION_f0_REFERENCE        = 0.33333d0

# attenuation period range over which we try to mimic a constant Q factor
MIN_ATTENUATION_PERIOD          = 999999998.d0
MAX_ATTENUATION_PERIOD          = 999999999.d0
# ignore this range and ask the code to compute it automatically instead based on the estimated resolution of the mesh (use this unless you know what you are doing)
COMPUTE_FREQ_BAND_AUTOMATIC     = .true.

# Olsen's constant for Q_mu = constant * V_s attenuation rule
USE_OLSEN_ATTENUATION           = .false.
OLSEN_ATTENUATION_RATIO         = 0.05

#-----------------------------------------------------------
#
# Absorbing boundary conditions
#
#-----------------------------------------------------------

# C-PML boundary conditions for a regional simulation
# (if set to .false., and STACEY_ABSORBING_CONDITIONS is also set to .false., you get a free surface instead
# in the case of elastic or viscoelastic mesh elements, and a rigid surface in the case of acoustic (fluid) elements
PML_CONDITIONS                  = .false.

# C-PML top surface
PML_INSTEAD_OF_FREE_SURFACE     = .false.

# C-PML dominant frequency
f0_FOR_PML                      = 0.05555

# absorbing boundary conditions for a regional simulation
# (if set to .false., and PML_CONDITIONS is also set to .false., you get a free surface instead
# in the case of elastic or viscoelastic mesh elements, and a rigid surface in the case of acoustic (fluid) elements
STACEY_ABSORBING_CONDITIONS     = .true.

# absorbing top surface (defined in mesh as 'free_surface_file')
STACEY_INSTEAD_OF_FREE_SURFACE  = .false.

# When STACEY_ABSORBING_CONDITIONS is set to .true. :
# absorbing conditions are defined in xmin, xmax, ymin, ymax and zmin
# this option BOTTOM_FREE_SURFACE can be set to .true. to
# make zmin free surface instead of absorbing condition
BOTTOM_FREE_SURFACE             = .false.

#-----------------------------------------------------------
#
# undoing attenuation and/or PMLs for sensitivity kernel calculations
#
#-----------------------------------------------------------

# to undo attenuation and/or PMLs for sensitivity kernel calculations or forward runs with SAVE_FORWARD
# use the flag below. It performs undoing of attenuation and/or of PMLs in an exact way for sensitivity kernel calculations
# but requires disk space for temporary storage, and uses a significant amount of memory used as buffers for temporary storage.
# When that option is on the second parameter indicates how often the code dumps restart files to disk (if in doubt, use something between 100 and 1000).
UNDO_ATTENUATION_AND_OR_PML     = .false.
NT_DUMP_ATTENUATION             = 500

#-----------------------------------------------------------
#
# Visualization
#
#-----------------------------------------------------------

# save AVS or OpenDX movies
# MOVIE_TYPE = 1 to show the top surface
# MOVIE_TYPE = 2 to show all the external faces of the mesh
CREATE_SHAKEMAP                 = .false.
MOVIE_SURFACE                   = .false.
MOVIE_TYPE                      = 1
MOVIE_VOLUME                    = .false.
SAVE_DISPLACEMENT               = .false.
MOVIE_VOLUME_TYPE               = 0
USE_HIGHRES_FOR_MOVIES          = .false.
NTSTEP_BETWEEN_FRAMES           = 200
HDUR_MOVIE                      = 0.0

# save AVS or OpenDX mesh files to check the mesh
SAVE_MESH_FILES                 = .false.

# path to store the local database file on each node
LOCAL_PATH                      = ./OUTPUT_FILES/DATABASES_MPI

# interval at which we output time step info and max of norm of displacement
NTSTEP_BETWEEN_OUTPUT_INFO      = 500

#-----------------------------------------------------------
#
# Sources
#
#-----------------------------------------------------------

# sources and receivers Z coordinates given directly (i.e. as their true position) instead of as their depth
USE_SOURCES_RECEIVERS_Z         = .false.

# use a (tilted) FORCESOLUTION force point source (or several) instead of a CMTSOLUTION moment-tensor source.
USE_FORCE_POINT_SOURCE          = .false.

# set to true to use a Ricker source time function instead of the source time functions set by default
# to represent a (tilted) FORCESOLUTION force point source or a CMTSOLUTION moment-tensor source.
USE_RICKER_TIME_FUNCTION        = .true.

# use an external source time function
# you must add a file with your source time function and the file name path
# relative to working directory at the end of CMTSOLUTION or FORCESOLUTION file
# (with multiple sources, one file per source is required).
USE_EXTERNAL_SOURCE_FILE        = .false.

# print source time function
PRINT_SOURCE_TIME_FUNCTION      = .false.

# source encoding
# (for acoustic simulations only for now) determines source encoding factor +/-1 depending on sign of moment tensor
USE_SOURCE_ENCODING             = .false.

#-----------------------------------------------------------
#
# Seismograms
#
#-----------------------------------------------------------

# interval in time steps for writing of seismograms
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = 10

# set to n to reduce the sampling rate of output seismograms by a factor of n
# defaults to 1, which means no down-sampling
NTSTEP_BETWEEN_OUTPUT_SAMPLE    = 1

# decide if we save displacement, velocity, acceleration and/or pressure in forward runs (they can be set to true simultaneously)
# currently pressure seismograms are implemented in acoustic (i.e. fluid) elements only
SAVE_SEISMOGRAMS_DISPLACEMENT   = .true.
SAVE_SEISMOGRAMS_VELOCITY       = .false.
SAVE_SEISMOGRAMS_ACCELERATION   = .false.
SAVE_SEISMOGRAMS_PRESSURE       = .false.

# option to save strain seismograms
# this option is useful for strain Green's tensor
SAVE_SEISMOGRAMS_STRAIN         = .false.

# save seismograms also when running the adjoint runs for an inverse problem
SAVE_SEISMOGRAMS_IN_ADJOINT_RUN = .false.

# save seismograms in binary or ASCII format (binary is smaller but may not be portable between machines)
USE_BINARY_FOR_SEISMOGRAMS      = .false.

# output seismograms in Seismic Unix format (binary with 240-byte-headers)
SU_FORMAT                       = .false.

# output seismograms in ASDF (requires asdf-library)
ASDF_FORMAT                     = .false.

# output seismograms in HDF5 (requires hdf5-library and WRITE_SEISMOGRAMS_BY_MAIN)
HDF5_FORMAT                     = .false.

# decide if main process writes all the seismograms or if all processes do it in parallel
WRITE_SEISMOGRAMS_BY_MAIN       = .false.

# save all seismograms in one large combined file instead of one file per seismogram
SAVE_ALL_SEISMOS_IN_ONE_FILE    = .false.

# use a trick to increase accuracy of pressure seismograms in fluid (acoustic) elements:
USE_TRICK_FOR_BETTER_PRESSURE   = .false.

#-----------------------------------------------------------
#
# Energy calculation
#
#-----------------------------------------------------------

# to plot energy curves, for instance to monitor how CPML absorbing layers behave;
# should be turned OFF in most cases because a bit expensive
OUTPUT_ENERGY                   = .false.
# every how many time steps we compute energy (which is a bit expensive to compute)
NTSTEP_BETWEEN_OUTPUT_ENERGY    = 10

#-----------------------------------------------------------
#
# Adjoint kernel outputs
#
#-----------------------------------------------------------

# interval in time steps for reading adjoint traces
# 0 = read the whole adjoint sources at start time
NTSTEP_BETWEEN_READ_ADJSRC      = 0

# read adjoint sources using ASDF (requires asdf-library)
READ_ADJSRC_ASDF                = .false.

# this parameter must be set to .true. to compute anisotropic kernels
# in crust and mantle (related to the 21 Cij in geographical coordinates)
# default is .false. to compute isotropic kernels (related to alpha and beta)
ANISOTROPIC_KL                  = .false.

# compute transverse isotropic kernels (alpha_v,alpha_h,beta_v,beta_h,eta,rho)
# rather than fully anisotropic kernels in case ANISOTROPIC_KL is set to .true.
SAVE_TRANSVERSE_KL              = .false.

# this parameter must be set to .true. to compute anisotropic kernels for
# cost function using velocity observable rather than displacement
ANISOTROPIC_VELOCITY_KL         = .false.

# outputs approximate Hessian for preconditioning
APPROXIMATE_HESS_KL             = .false.

# save Moho mesh and compute Moho boundary kernels
SAVE_MOHO_MESH                  = .false.

#-----------------------------------------------------------
#
# Coupling with an injection technique (DSM, AxiSEM, or FK)
#
#-----------------------------------------------------------
COUPLE_WITH_INJECTION_TECHNIQUE = .false.
INJECTION_TECHNIQUE_TYPE        = 3   # 1 = DSM, 2 = AxiSEM, 3 = FK
MESH_A_CHUNK_OF_THE_EARTH       = .false.
TRACTION_PATH                   = ./DATA/AxiSEM_tractions/3/
FKMODEL_FILE                    = FKmodel
RECIPROCITY_AND_KH_INTEGRAL     = .false.   # does not work yet

#-----------------------------------------------------------
#
# Run modes
#
#-----------------------------------------------------------

# Simultaneous runs
NUMBER_OF_SIMULTANEOUS_RUNS     = 1

# if we perform simultaneous runs in parallel, if only the source and receivers vary between these runs
# but not the mesh nor the model (velocity and density) then we can also read the mesh and model files
# from a single run in the beginning and broadcast them to all the others
BROADCAST_SAME_MESH_AND_MODEL   = .true.

#-----------------------------------------------------------

# set to true to use GPUs
GPU_MODE                        = .false.

# ADIOS Options for I/Os
ADIOS_ENABLED                   = .false.
ADIOS_FOR_DATABASES             = .false.
ADIOS_FOR_MESH                  = .false.
ADIOS_FOR_FORWARD_ARRAYS        = .false.
ADIOS_FOR_KERNELS               = .false.
ADIOS_FOR_UNDO_ATTENUATION      = .false.

# HDF5 Database I/O
# (note the flags for HDF5 and ADIOS are mutually exclusive, only one can be used)
HDF5_ENABLED                    = .false.
HDF5_FOR_MOVIES                 = .false.
HDF5_IO_NODES                   = 0   # HDF5 IO server with number of IO dedicated procs
""")

# Create a compatible Mesh_Par_file
mesh_par_file = os.path.join(MESH_DIR, "Mesh_Par_file")
print(f"Creating compatible Mesh_Par_file at {mesh_par_file}")

with open(mesh_par_file, 'w') as f:
    f.write("""#-----------------------------------------------------------
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

# UTM projection parameters
UTM_PROJECTION_ZONE             = 11
SUPPRESS_UTM_PROJECTION         = .true.

# file that contains the interfaces of the model / mesh
INTERFACES_FILE                 = interfaces.dat

# file that contains the cavity
CAVITY_FILE                     = no_cavity.dat

# number of elements at the surface along edges of the mesh at the surface
# (must be 8 * multiple of NPROC below if mesh is not regular and contains mesh doublings)
# (must be multiple of NPROC below if mesh is regular)
NEX_XI                          = 40
NEX_ETA                         = 40

# number of MPI processors along xi and eta (can be different)
NPROC_XI                        = 2
NPROC_ETA                       = 2

#-----------------------------------------------------------
#
# Doubling layers
#
#-----------------------------------------------------------

# Regular/irregular mesh
USE_REGULAR_MESH                = .true.
# Only for irregular meshes, number of doubling layers and their position
NDOUBLINGS                      = 0
# NZ_DOUBLING_1 is the parameter to set up if there is only one doubling layer
# (more doubling entries can be added if needed to match NDOUBLINGS value)
NZ_DOUBLING_1                   = 0
NZ_DOUBLING_2                   = 0

#-----------------------------------------------------------
#
# Visualization
#
#-----------------------------------------------------------

# create mesh files for visualisation or further checking
CREATE_ABAQUS_FILES             = .false.
CREATE_DX_FILES                 = .false.
CREATE_VTK_FILES                = .true.

# path to store the databases files
LOCAL_PATH                      = ./DATABASES_MPI

#-----------------------------------------------------------
#
# Domain materials
#
#-----------------------------------------------------------

# number of materials
NMATERIALS                      = 1
# define the different materials in the model as:
# #material_id  #rho  #vp  #vs  #Q_Kappa  #Q_mu  #anisotropy_flag  #domain_id
1 2700.0 3500.0 2000.0 9999.0 9999.0 0 2

#-----------------------------------------------------------
#
# Domain regions
#
#-----------------------------------------------------------

# number of regions
NREGIONS                        = 1
# define the different regions of the model as :
#NEX_XI_BEGIN  #NEX_XI_END  #NEX_ETA_BEGIN  #NEX_ETA_END  #NZ_BEGIN #NZ_END  #material_id
1 40 1 40 1 40 1
""")

# Also copy Mesh_Par_file to DATA directory
shutil.copy(mesh_par_file, os.path.join(DATA_DIR, "Mesh_Par_file"))

# Create interface and cavity files
interfaces_file = os.path.join(MESH_DIR, "interfaces.dat")
cavity_file = os.path.join(MESH_DIR, "no_cavity.dat")

print(f"Creating interfaces.dat and no_cavity.dat")

with open(interfaces_file, 'w') as f:
    f.write("""# number of interfaces
1
# for each interface below, we give the number of points in xi and eta directions
2 2
# and then x,y,z for all these points (xi and eta being the first and second directions respectively)
0.0 0.0 0.0
2000.0 0.0 0.0
0.0 2000.0 0.0
2000.0 2000.0 0.0
""")

with open(cavity_file, 'w') as f:
    f.write("# No cavity")

# Create CMTSOLUTION file
cmt_file = os.path.join(DATA_DIR, "CMTSOLUTION")
print(f"Creating CMTSOLUTION file")

with open(cmt_file, 'w') as f:
    f.write("""PDE 2000 1 1 0 0 0.0 1000.0 1000.0 500.0 4.5 0.0 0.0 sample_event
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
""")

# Create stations file
stations_file = os.path.join(DATA_DIR, "STATIONS")
print(f"Creating STATIONS file")

with open(stations_file, 'w') as f:
    for i in range(1, 101):
        x = 500.0 + (1000.0 * (i-1) / 99)
        f.write(f"ST{i:03d} GE {x} 1000.0 0.0 0.0\n")

# Copy stations file to project output directory
shutil.copy(stations_file, os.path.join(SIMULATION_DIR, "STATIONS"))

# Create SOURCE file
source_file = os.path.join(DATA_DIR, "SOURCE")
print(f"Creating SOURCE file")

with open(source_file, 'w') as f:
    f.write("""source_surf = 0
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
""")

# Copy source file to project output directory
shutil.copy(source_file, os.path.join(SIMULATION_DIR, "SOURCE"))

# Copy Par_file to project output directory
shutil.copy(par_file, os.path.join(SIMULATION_DIR, "Par_file"))

print("\nAll configuration files have been created with full v4.1.1 parameter compliance.")
print("Now fixing the UTM_PROJECTION_ZONE parameter format issue...")

# Fix the UTM_PROJECTION_ZONE format issue
def fix_utm_zone():
    files_to_check = [
        os.path.join(DATA_DIR, "Par_file"),
        os.path.join(DATA_DIR, "Mesh_Par_file"),
        os.path.join(MESH_DIR, "Mesh_Par_file")
    ]
    
    for file_path in files_to_check:
        # Read the file
        try:
            with open(file_path, 'r') as f:
                content = f.readlines()
            
            # Modify the UTM_PROJECTION_ZONE line
            for i, line in enumerate(content):
                if line.strip().startswith("UTM_PROJECTION_ZONE"):
                    content[i] = "UTM_PROJECTION_ZONE             = 11\n"
            
            # Write the file back
            with open(file_path, 'w') as f:
                f.writelines(content)
            
            print(f"Fixed UTM_PROJECTION_ZONE in {file_path}")
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

# Execute the fix
fix_utm_zone()

print("""
You can now run SPECFEM3D with:
cd ~/specfem3d
mpirun -np 4 ./bin/xmeshfem3D
mpirun -np 4 ./bin/xdecompose_mesh
mpirun -np 4 ./bin/xspecfem3D
""")