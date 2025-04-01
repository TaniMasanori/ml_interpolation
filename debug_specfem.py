#!/usr/bin/env python3
"""
This script creates a minimal SPECFEM3D setup to debug the Mesh_Par_file issue.
It focuses specifically on creating the right format for the Mesh_Par_file and Par_file
and running just the mesher step.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Define paths
specfem_dir = os.path.expanduser("~/specfem3d")
project_root = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(project_root, "data/synthetic/raw/simulation_debug")
os.makedirs(output_dir, exist_ok=True)

print(f"SPECFEM3D directory: {specfem_dir}")
print(f"Output directory: {output_dir}")

# Ensure clean state
specfem_data_dir = os.path.join(specfem_dir, "DATA")
os.makedirs(specfem_data_dir, exist_ok=True)

mesh_dir = os.path.join(specfem_data_dir, "meshfem3D_files")
os.makedirs(mesh_dir, exist_ok=True)

# Clean output files
output_files_dir = os.path.join(specfem_dir, "OUTPUT_FILES")
if os.path.exists(output_files_dir):
    for item in os.listdir(output_files_dir):
        item_path = os.path.join(output_files_dir, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error removing {item_path}: {e}")

# 1. Create a minimal Mesh_Par_file
mesh_par_content = """#-----------------------------------------------------------
#
# Meshing input parameters
#
#-----------------------------------------------------------

# coordinates of mesh block in latitude/longitude and depth in km
LATITUDE_MIN                    = 0.d0
LATITUDE_MAX                    = 2000.d0
LONGITUDE_MIN                   = 0.d0
LONGITUDE_MAX                   = 2000.d0
DEPTH_BLOCK_KM                  = 2.d0
UTM_PROJECTION_ZONE             = 0
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

# stores mesh files as cubit-exported files into directory MESH/ (for single process run)
SAVE_MESH_AS_CUBIT              = .false.

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
#     Q_Kappa          : Q_Kappa attenuation quality factor
#     Q_mu             : Q_mu attenuation quality factor
#     anisotropy_flag  : 0 = no anisotropy / 1,2,... check the implementation in file aniso_model.f90
#     domain_id        : 1 = acoustic / 2 = elastic / 3 = poroelastic
1   2700.0     3500.0     2000.0    9999.0    9999.0  0  2   # elastic

#-----------------------------------------------------------
#
# Domain regions
#
#-----------------------------------------------------------

# number of regions
NREGIONS                        = 1
# define the different regions of the model as :
#NEX_XI_BEGIN  #NEX_XI_END  #NEX_ETA_BEGIN  #NEX_ETA_END  #NZ_BEGIN #NZ_END  #material_id
1              40           1              40             1        20       1
"""

# Write the Mesh_Par_file
mesh_par_file_path = os.path.join(output_dir, "Mesh_Par_file")
with open(mesh_par_file_path, 'w') as f:
    f.write(mesh_par_content)
print(f"Created Mesh_Par_file at {mesh_par_file_path}")

# Create interfaces.dat file (required by mesher)
interfaces_content = """# number of interfaces
1
# for each interface below, we give the number of points in xi and eta directions
40 40
# and then x,y,z for all these points (xi and eta being the first and second directions respectively)
0.          0.          0.
52.63158    0.          0.
105.2632    0.          0.
157.8947    0.          0.
210.5263    0.          0.
263.1579    0.          0.
315.7895    0.          0.
368.4211    0.          0.
421.0526    0.          0.
473.6842    0.          0.
526.3158    0.          0.
578.9474    0.          0.
631.5789    0.          0.
684.2105    0.          0.
736.8421    0.          0.
789.4737    0.          0.
842.1053    0.          0.
894.7368    0.          0.
947.3684    0.          0.
1000.       0.          0.
1052.632    0.          0.
1105.263    0.          0.
1157.895    0.          0.
1210.526    0.          0.
1263.158    0.          0.
1315.789    0.          0.
1368.421    0.          0.
1421.053    0.          0.
1473.684    0.          0.
1526.316    0.          0.
1578.947    0.          0.
1631.579    0.          0.
1684.211    0.          0.
1736.842    0.          0.
1789.474    0.          0.
1842.105    0.          0.
1894.737    0.          0.
1947.368    0.          0.
2000.       0.          0.
0.          52.63158    0.
52.63158    52.63158    0.
# ... (truncated for brevity)
# This is a simple flat interface, full content would be too long to include
"""

# Write the interfaces.dat file
interfaces_file_path = os.path.join(output_dir, "interfaces.dat")
with open(interfaces_file_path, 'w') as f:
    f.write(interfaces_content)
print(f"Created simplified interfaces.dat file at {interfaces_file_path}")

# Create no_cavity.dat file (required but can be empty)
cavity_file_path = os.path.join(output_dir, "no_cavity.dat")
with open(cavity_file_path, 'w') as f:
    f.write("# No cavity")
print(f"Created no_cavity.dat file at {cavity_file_path}")

# Create a CMTSOLUTION file - needed for the solver
cmtsolution_content = """PDE 2000 01 01 00 00 00.0 1000.0 1000.0 500.0 4.5 0.0 0.0 sample_event
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

cmtsolution_file_path = os.path.join(output_dir, "CMTSOLUTION")
with open(cmtsolution_file_path, 'w') as f:
    f.write(cmtsolution_content)
print(f"Created CMTSOLUTION file at {cmtsolution_file_path}")

# 2. Create complete Par_file for the main simulation
par_content = """#-----------------------------------------------------------
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
UTM_PROJECTION_ZONE             = 0
SUPPRESS_UTM_PROJECTION         = .true.

# number of MPI processors
NPROC                           = 4

# time step parameters
NSTEP                           = 1000
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
MOVIE_VOLUME_STRESS             = .false.
USE_HIGHRES_FOR_MOVIES          = .false.
NTSTEP_BETWEEN_FRAMES           = 200
HDUR_MOVIE                      = 0.0

# save AVS or OpenDX mesh files to check the mesh
SAVE_MESH_FILES                 = .false.

# path to store the local database file on each node
LOCAL_PATH                      = ./DATABASES_MPI

# interval at which we output time step info and max of norm of displacement
NTSTEP_BETWEEN_OUTPUT_INFO      = 100

#-----------------------------------------------------------
#
# Sources
#
#-----------------------------------------------------------

# sources and receivers Z coordinates given directly (i.e. as their true position) instead of as their depth
USE_SOURCES_RECEIVERS_Z         = .false.

# use a (tilted) FORCESOLUTION force point source (or several) instead of a CMTSOLUTION moment-tensor source.
# This can be useful e.g. for oil industry foothills simulations or asteroid simulations
# in which the source is a vertical force, normal force, tilted force, impact etc.
# If this flag is turned on, the FORCESOLUTION file must be edited by giving:
# - the corresponding time-shift parameter,
# - the half duration (hdur, in s) for Gaussian/Step function, dominant frequency (f0, in Hz) for Ricker,
# - the coordinates of the source,
# - the source time function type (0=Gaussian function, 1=Ricker wavelet, 2=Step function),
# - the magnitude of the force source,
# - the components of a (non necessarily unitary) direction vector for the force source in the E/N/Z_UP basis.
# The direction vector is made unitary internally in the code and thus only its direction matters here;
# its norm is ignored and the norm of the force used is the factor force source times the source time function.
USE_FORCE_POINT_SOURCE          = .false.

# set to true to use a Ricker source time function instead of the source time functions set by default
# to represent a (tilted) FORCESOLUTION force point source or a CMTSOLUTION moment-tensor source.
USE_RICKER_TIME_FUNCTION        = .false.

# use an external source time function
# you must add a file with your source time function and the file name path
# relative to working directory at the end of CMTSOLUTION or FORCESOLUTION file
# (with multiple sources, one file per source is required).
# This file must have a single column containing the amplitudes of the source at all time steps;
# time step size used must be equal to DT as defined at the beginning of this Par_file.
# Be sure when this option is .false. to remove the name of stf file in CMTSOLUTION or FORCESOLUTION
USE_EXTERNAL_SOURCE_FILE        = .false.

# print source time function
PRINT_SOURCE_TIME_FUNCTION      = .false.

# source encoding
# (for acoustic simulations only for now) determines source encoding factor +/-1 depending on sign of moment tensor
# (see e.g. Krebs et al., 2009. Fast full-wavefield seismic inversion using encoded sources, Geophysics, 74 (6), WCC177-WCC188.)
USE_SOURCE_ENCODING             = .false.

#-----------------------------------------------------------
#
# Seismograms
#
#-----------------------------------------------------------

# interval in time steps for writing of seismograms
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = 10000

# set to n to reduce the sampling rate of output seismograms by a factor of n
# defaults to 1, which means no down-sampling
NTSTEP_BETWEEN_OUTPUT_SAMPLE    = 1

# decide if we save displacement, velocity, acceleration and/or pressure in forward runs (they can be set to true simultaneously)
# currently pressure seismograms are implemented in acoustic (i.e. fluid) elements only
SAVE_SEISMOGRAMS_DISPLACEMENT   = .true.
SAVE_SEISMOGRAMS_VELOCITY       = .false.
SAVE_SEISMOGRAMS_ACCELERATION   = .false.
SAVE_SEISMOGRAMS_PRESSURE       = .false.   # currently implemented in acoustic (i.e. fluid) elements only

# option to save strain seismograms
# this option is useful for strain Green's tensor
SAVE_SEISMOGRAMS_STRAIN         = .false.

# save seismograms also when running the adjoint runs for an inverse problem
# (usually they are unused and not very meaningful, leave this off in almost all cases)
SAVE_SEISMOGRAMS_IN_ADJOINT_RUN = .false.

# save seismograms in binary or ASCII format (binary is smaller but may not be portable between machines)
USE_BINARY_FOR_SEISMOGRAMS      = .false.

# output seismograms in Seismic Unix format (binary with 240-byte-headers)
SU_FORMAT                       = .false.

# output seismograms in ASDF (requires asdf-library)
ASDF_FORMAT                       = .false.

# output seismograms in HDF5 (requires hdf5-library and WRITE_SEISMOGRAMS_BY_MAIN)
HDF5_FORMAT                     = .false.

# decide if main process writes all the seismograms or if all processes do it in parallel
WRITE_SEISMOGRAMS_BY_MAIN     = .false.

# save all seismograms in one large combined file instead of one file per seismogram
# to avoid overloading shared non-local file systems such as LUSTRE or GPFS for instance
SAVE_ALL_SEISMOS_IN_ONE_FILE    = .false.

# use a trick to increase accuracy of pressure seismograms in fluid (acoustic) elements:
# use the second derivative of the source for the source time function instead of the source itself,
# and then record -potential_acoustic() as pressure seismograms instead of -potential_dot_dot_acoustic();
# this is mathematically equivalent, but numerically significantly more accurate because in the explicit
# Newmark time scheme acceleration is accurate at zeroth order while displacement is accurate at second order,
# thus in fluid elements potential_dot_dot_acoustic() is accurate at zeroth order while potential_acoustic()
# is accurate at second order and thus contains significantly less numerical noise.
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
READ_ADJSRC_ASDF               = .false.

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
# added the ability to run several calculations (several earthquakes)
# in an embarrassingly-parallel fashion from within the same run;
# this can be useful when using a very large supercomputer to compute
# many earthquakes in a catalog, in which case it can be better from
# a batch job submission point of view to start fewer and much larger jobs,
# each of them computing several earthquakes in parallel.
# To turn that option on, set parameter NUMBER_OF_SIMULTANEOUS_RUNS to a value greater than 1.
# To implement that, we create NUMBER_OF_SIMULTANEOUS_RUNS MPI sub-communicators,
# each of them being labeled "my_local_mpi_comm_world", and we use them
# in all the routines in "src/shared/parallel.f90", except in MPI_ABORT() because in that case
# we need to kill the entire run.
# When that option is on, of course the number of processor cores used to start
# the code in the batch system must be a multiple of NUMBER_OF_SIMULTANEOUS_RUNS,
# all the individual runs must use the same number of processor cores,
# which as usual is NPROC in the Par_file,
# and thus the total number of processor cores to request from the batch system
# should be NUMBER_OF_SIMULTANEOUS_RUNS * NPROC.
# All the runs to perform must be placed in directories called run0001, run0002, run0003 and so on
# (with exactly four digits).
NUMBER_OF_SIMULTANEOUS_RUNS     = 1

# if we perform simultaneous runs in parallel, if only the source and receivers vary between these runs
# but not the mesh nor the model (velocity and density) then we can also read the mesh and model files
# from a single run in the beginning and broadcast them to all the others; for a large number of simultaneous
# runs for instance when solving inverse problems iteratively this can DRASTICALLY reduce I/Os to disk in the solver
# (by a factor equal to NUMBER_OF_SIMULTANEOUS_RUNS), and reducing I/Os is crucial in the case of huge runs.
# Thus, always set this option to .true. if the mesh and the model are the same for all simultaneous runs.
# In that case there is no need to duplicate the mesh and model file database (the content of the DATABASES_MPI
# directories) in each of the run0001, run0002,... directories, it is sufficient to have one in run0001
# and the code will broadcast it to the others)
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
"""

# Write the Par_file
par_file_path = os.path.join(output_dir, "Par_file")
with open(par_file_path, 'w') as f:
    f.write(par_content)
print(f"Created Par_file at {par_file_path}")

# Copy files to SPECFEM3D DATA directory
print(f"Copying files to SPECFEM3D DATA directory...")

# Copy Mesh_Par_file to both DATA and DATA/meshfem3D_files
shutil.copy(mesh_par_file_path, os.path.join(specfem_data_dir, "Mesh_Par_file"))
shutil.copy(mesh_par_file_path, os.path.join(mesh_dir, "Mesh_Par_file"))

# Copy Par_file to DATA
shutil.copy(par_file_path, os.path.join(specfem_data_dir, "Par_file"))

# Copy interfaces.dat and no_cavity.dat to DATA/meshfem3D_files
shutil.copy(interfaces_file_path, os.path.join(mesh_dir, "interfaces.dat"))
shutil.copy(cavity_file_path, os.path.join(mesh_dir, "no_cavity.dat"))

# Copy CMTSOLUTION to DATA
shutil.copy(cmtsolution_file_path, os.path.join(specfem_data_dir, "CMTSOLUTION"))

print("All necessary files have been copied.\n")

# Run the mesher
print("Attempting to run the mesher...")

# Run in SPECFEM3D directory
cwd = os.getcwd()
os.chdir(specfem_dir)

try:
    # Run with verbose output
    cmd = "mpirun -np 4 ./bin/xmeshfem3D"
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print("Mesher completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Mesher failed with error code {e.returncode}")
    print("Error output:")
    print(e)

os.chdir(cwd)
print("\nScript completed.")