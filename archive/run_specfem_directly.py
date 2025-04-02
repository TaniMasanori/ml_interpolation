#!/usr/bin/env python3
"""
This script runs SPECFEM3D directly using the example files from the EXAMPLES directory.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Define paths
specfem_dir = os.path.expanduser("~/specfem3d")
project_root = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(project_root, "data/synthetic/raw/simulation_direct")
os.makedirs(output_dir, exist_ok=True)

print(f"SPECFEM3D directory: {specfem_dir}")
print(f"Output directory: {output_dir}")

# 1. Clean the DATA and OUTPUT_FILES directories in SPECFEM3D
print("Cleaning SPECFEM3D directories...")
data_dir = os.path.join(specfem_dir, "DATA")
output_files_dir = os.path.join(specfem_dir, "OUTPUT_FILES")

# Clean output files
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

# Create the OUTPUT_FILES/DATABASES_MPI directory if it doesn't exist
os.makedirs(os.path.join(output_files_dir, "DATABASES_MPI"), exist_ok=True)

# 2. Copy files from an example
example_dir = os.path.join(specfem_dir, "EXAMPLES/applications/CPML_examples/homogeneous_halfspace_HEX8_elastic_absorbing_CPML_5sides")
example_data_dir = os.path.join(example_dir, "DATA")

if not os.path.exists(example_data_dir):
    print(f"Error: Example data directory {example_data_dir} not found!")
    # List available examples
    print("Available examples:")
    subprocess.run(f"find {specfem_dir}/EXAMPLES -name DATA -type d | head -10", shell=True)
    sys.exit(1)

print(f"Copying files from example: {example_data_dir}")

# Copy all files from example DATA directory to SPECFEM3D DATA directory
for item in os.listdir(example_data_dir):
    src = os.path.join(example_data_dir, item)
    dst = os.path.join(data_dir, item)
    
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"  Copied file: {item}")
    elif os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  Copied directory: {item}")

# 3. Run SPECFEM3D steps
print("\nRunning SPECFEM3D steps directly from the example...")

# Get NPROC from Par_file
nproc = 4  # Default
par_file_path = os.path.join(data_dir, "Par_file")
if os.path.exists(par_file_path):
    with open(par_file_path, 'r') as f:
        for line in f:
            if "NPROC" in line and "=" in line and not line.strip().startswith("#"):
                try:
                    nproc = int(line.split("=")[1].strip())
                    print(f"Found NPROC = {nproc} in Par_file")
                    break
                except:
                    pass

# Run in SPECFEM3D directory
cwd = os.getcwd()
os.chdir(specfem_dir)

# Step 1: Run the mesher
print("\n=== Step 1: Running mesher ===")
cmd = f"mpirun -np {nproc} ./bin/xmeshfem3D"
print(f"Executing: {cmd}")
result = os.system(cmd)
if result != 0:
    print("⚠️ Meshing failed")
    os.chdir(cwd)
    sys.exit(1)
print("✅ Meshing completed successfully")

# Step 2: Run the partitioner
print("\n=== Step 2: Running partitioner ===")
cmd = f"mpirun -np {nproc} ./bin/xdecompose_mesh"
print(f"Executing: {cmd}")
result = os.system(cmd)
if result != 0:
    print("⚠️ Partitioning failed")
    os.chdir(cwd)
    sys.exit(1)
print("✅ Partitioning completed successfully")

# Step 3: Run the solver
print("\n=== Step 3: Running solver ===")
cmd = f"mpirun -np {nproc} ./bin/xspecfem3D"
print(f"Executing: {cmd}")
result = os.system(cmd)
if result != 0:
    print("⚠️ Solver failed")
    os.chdir(cwd)
    sys.exit(1)
print("✅ Solver completed successfully")

# 4. Copy output files back to the project directory
print("\nCopying output files to project directory...")
seismo_count = 0
for file in os.listdir(output_files_dir):
    if file.endswith(".semd") or file.endswith(".semp") or file.endswith(".sema"):
        src_file = os.path.join(output_files_dir, file)
        dst_file = os.path.join(output_dir, file)
        shutil.copy(src_file, dst_file)
        seismo_count += 1

print(f"Copied {seismo_count} seismogram files to {output_dir}")

# Return to original directory
os.chdir(cwd)
print("\nDone!")