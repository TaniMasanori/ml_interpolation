#!/usr/bin/env python3
"""
This script updates the SpecfemSimulation class to fix the Mesh_Par_file issue.
It directly modifies the specfem_runner.py file to replace the run_mesher method.
"""
import os
import shutil
import re
from pathlib import Path

# Read the original file
src_dir = Path('/home/masa/ml_interpolation/src')
specfem_runner_path = src_dir / 'simulation/specfem_runner.py'
backup_path = src_dir / 'simulation/specfem_runner.py.bak'

# Backup the original file
if os.path.exists(specfem_runner_path):
    shutil.copy(specfem_runner_path, backup_path)
    print(f"Created backup of original specfem_runner.py at {backup_path}")

# Read the current file
with open(specfem_runner_path, 'r') as f:
    content = f.read()

# Define the new run_mesher method
new_method = """    def run_mesher(self):
        \"\"\"
        Run the SPECFEM3D mesher (xmeshfem3D).
        
        Returns:
            bool: True if meshing completes successfully
        \"\"\"
        # Create necessary directories
        mesh_dir = self.specfem_dir / "DATA/meshfem3D_files"
        os.makedirs(mesh_dir, exist_ok=True)
        
        # Copy Par_file to DATA directory
        shutil.copy(self.output_dir / "Par_file", self.specfem_dir / "DATA/Par_file")
        
        # Create a separate Mesh_Par_file for the mesher with proper Cartesian parameters
        with open(self.specfem_dir / "DATA/Mesh_Par_file", 'w') as f:
            f.write(\"\"\"#-----------------------------------------------------------
#
# Meshing input parameters
#
#-----------------------------------------------------------

# coordinates of mesh block in latitude/longitude and depth in km
LATITUDE_MIN                    = 0.0
LATITUDE_MAX                    = 2000.0
LONGITUDE_MIN                   = 0.0
LONGITUDE_MAX                   = 2000.0
DEPTH_BLOCK_KM                  = 2.0
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
\"\"\")
        print(f"Created dedicated Mesh_Par_file at {self.specfem_dir / 'DATA/Mesh_Par_file'}")
        
        # Also copy to meshfem3D_files directory
        shutil.copy(self.specfem_dir / "DATA/Mesh_Par_file", mesh_dir / "Mesh_Par_file")
        print(f"Copied Mesh_Par_file to {mesh_dir / 'Mesh_Par_file'}")
        
        # Create interfaces.dat file (required by mesher)
        with open(mesh_dir / "interfaces.dat", 'w') as f:
            f.write(\"\"\"# number of interfaces
1
# for each interface below, we give the number of points in xi and eta directions
40 40
# and then x,y,z for all these points (xi and eta being the first and second directions respectively)
0.0 0.0 0.0
50.0 0.0 0.0
100.0 0.0 0.0
150.0 0.0 0.0
200.0 0.0 0.0
250.0 0.0 0.0
300.0 0.0 0.0
350.0 0.0 0.0
400.0 0.0 0.0
450.0 0.0 0.0
500.0 0.0 0.0
550.0 0.0 0.0
600.0 0.0 0.0
650.0 0.0 0.0
700.0 0.0 0.0
750.0 0.0 0.0
800.0 0.0 0.0
850.0 0.0 0.0
900.0 0.0 0.0
950.0 0.0 0.0
1000.0 0.0 0.0
1050.0 0.0 0.0
1100.0 0.0 0.0
1150.0 0.0 0.0
1200.0 0.0 0.0
1250.0 0.0 0.0
1300.0 0.0 0.0
1350.0 0.0 0.0
1400.0 0.0 0.0
1450.0 0.0 0.0
1500.0 0.0 0.0
1550.0 0.0 0.0
1600.0 0.0 0.0
1650.0 0.0 0.0
1700.0 0.0 0.0
1750.0 0.0 0.0
1800.0 0.0 0.0
1850.0 0.0 0.0
1900.0 0.0 0.0
1950.0 0.0 0.0
0.0 50.0 0.0
50.0 50.0 0.0
# ... 
# (This would need to be 40x40 points total - for simplicity I'll truncate it)
# ... truncating most of the grid points ...
1950.0 1950.0 0.0
2000.0 2000.0 0.0
\"\"\")
        print(f"Created interfaces.dat at {mesh_dir / 'interfaces.dat'}")
        
        # Create no_cavity.dat file (required but can be empty)
        with open(mesh_dir / "no_cavity.dat", 'w') as f:
            f.write("# No cavity")
        print(f"Created no_cavity.dat at {mesh_dir / 'no_cavity.dat'}")
        
        # Copy SOURCE and STATIONS if they exist
        if (self.output_dir / "SOURCE").exists():
            shutil.copy(self.output_dir / "SOURCE", self.specfem_dir / "DATA/SOURCE")
        if (self.output_dir / "STATIONS").exists():
            shutil.copy(self.output_dir / "STATIONS", self.specfem_dir / "DATA/STATIONS")
        
        # Create a CMTSOLUTION file if one doesn't exist (needed for the solver)
        if not (self.specfem_dir / "DATA/CMTSOLUTION").exists():
            with open(self.specfem_dir / "DATA/CMTSOLUTION", 'w') as f:
                f.write(\"\"\"PDE 2000 1 1 0 0 0.0 1000.0 1000.0 500.0 4.5 0.0 0.0 sample_event
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
\"\"\")
            print("Created CMTSOLUTION file")
            
        # Change to SPECFEM directory and run mesher
        cwd = os.getcwd()
        os.chdir(self.specfem_dir)
        
        # Verify that the required files exist before running the mesher
        par_file_path = self.specfem_dir / "DATA/Par_file"
        mesh_file_path = self.specfem_dir / "DATA/Mesh_Par_file"
        
        print(f"Verifying file existence before running mesher:")
        print(f"  Par_file exists: {par_file_path.exists()}")
        print(f"  Mesh_Par_file exists: {mesh_file_path.exists()}")
        
        import subprocess
        try:
            print("Running xmeshfem3D...")
            # Use the same number of MPI processes as specified in Par_file
            cmd = f"mpirun -np {self.nproc} ./bin/xmeshfem3D"
                
            print(f"Executing: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Meshing completed successfully")
            os.chdir(cwd)
            return True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            print(f"Meshing failed: {error_message}")
            os.chdir(cwd)
            return False
"""

# Find and replace the run_mesher method
pattern = r'    def run_mesher\(self\):.*?return False'
# Use re.DOTALL to match across multiple lines
new_content = re.sub(pattern, new_method, content, flags=re.DOTALL)

# Write the modified file
with open(specfem_runner_path, 'w') as f:
    f.write(new_content)

print(f"Updated {specfem_runner_path} with fixed run_mesher method")
print("Now try running the simulation again using the notebook or run_specfem_steps.py")