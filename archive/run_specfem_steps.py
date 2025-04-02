#!/usr/bin/env python3
"""
Run SPECFEM3D simulation step by step from within ml_interpolation directory.
This script first runs the fix_specfem_v4.1.1.py script to generate all necessary
configuration files, then runs the full SPECFEM3D simulation.
"""
import os
import subprocess
import sys
import time

# Step 0: Run the parameter fix script first
print("\n=== Step 0: Running the parameter fix script ===")
fix_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_specfem_v4.1.1.py")
print(f"Executing: {fix_script}")

try:
    process = subprocess.Popen([sys.executable, fix_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Print output
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    
    print("\nSTDOUT:")
    print(stdout_str)
    
    if stderr_str:
        print("\nSTDERR:")
        print(stderr_str)
    
    if process.returncode != 0:
        print(f"Parameter fix script failed with return code {process.returncode}")
        exit(1)
    else:
        print("Parameter fix script completed successfully")
except Exception as e:
    print(f"Error running parameter fix script: {e}")
    exit(1)

# Define paths
SPECFEM_DIR = os.path.expanduser("~/specfem3d")
OUTPUT_DIR = os.path.join(SPECFEM_DIR, "OUTPUT_FILES")
SIMULATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "data/synthetic/raw/simulation1")

print(f"\n=== Running SPECFEM3D Simulation ===")
print(f"SPECFEM3D directory: {SPECFEM_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Project simulation directory: {SIMULATION_DIR}")

# Give a short pause to ensure file system is updated
time.sleep(1)

# Step 1: Run the mesher
print("\n=== Step 1: Running the mesher (xmeshfem3D) ===")

# We can't change directory in this environment, so use absolute paths
cmd = f"mpirun -np 4 {SPECFEM_DIR}/bin/xmeshfem3D"
print(f"Executing: {cmd}")

try:
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=SPECFEM_DIR)
    stdout, stderr = process.communicate()
    
    # Print output
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    
    print("\nSTDOUT:")
    print(stdout_str)
    
    if stderr_str:
        print("\nSTDERR:")
        print(stderr_str)
    
    if process.returncode != 0:
        print(f"Mesher failed with return code {process.returncode}")
        exit(1)
    else:
        print("Mesher completed successfully")
except Exception as e:
    print(f"Error running mesher: {e}")
    exit(1)

# Step 2: Run the partitioner
print("\n=== Step 2: Running the partitioner (xdecompose_mesh) ===")
cmd = f"mpirun -np 4 {SPECFEM_DIR}/bin/xdecompose_mesh"
print(f"Executing: {cmd}")

try:
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=SPECFEM_DIR)
    stdout, stderr = process.communicate()
    
    # Print output
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    
    print("\nSTDOUT:")
    print(stdout_str)
    
    if stderr_str:
        print("\nSTDERR:")
        print(stderr_str)
    
    if process.returncode != 0:
        print(f"Partitioner failed with return code {process.returncode}")
        exit(1)
    else:
        print("Partitioner completed successfully")
except Exception as e:
    print(f"Error running partitioner: {e}")
    exit(1)

# Step 3: Run the solver
print("\n=== Step 3: Running the solver (xspecfem3D) ===")
cmd = f"mpirun -np 4 {SPECFEM_DIR}/bin/xspecfem3D"
print(f"Executing: {cmd}")

try:
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=SPECFEM_DIR)
    stdout, stderr = process.communicate()
    
    # Print output
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    
    print("\nSTDOUT:")
    print(stdout_str)
    
    if stderr_str:
        print("\nSTDERR:")
        print(stderr_str)
    
    if process.returncode != 0:
        print(f"Solver failed with return code {process.returncode}")
        exit(1)
    else:
        print("Solver completed successfully")
except Exception as e:
    print(f"Error running solver: {e}")
    exit(1)

# Step 4: Copy output files to simulation directory
print("\n=== Step 4: Copying output files to simulation directory ===")
try:
    # Instead of directly copying files, we'll use rsync through subprocess
    cmd = f"rsync -av --include='*.semd' --include='*.semp' --include='*.sema' --exclude='*' {OUTPUT_DIR}/ {SIMULATION_DIR}/"
    print(f"Executing: {cmd}")
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    
    print("\nSTDOUT:")
    print(stdout_str)
    
    if stderr_str:
        print("\nSTDERR:")
        print(stderr_str)
    
    if process.returncode != 0:
        print(f"Copying files failed with return code {process.returncode}")
    else:
        print("Successfully copied output files")
except Exception as e:
    print(f"Error copying output files: {e}")
    print("Continuing anyway...")

print("\n=== Simulation Completed Successfully ===")
print(f"Output files are available in:")
print(f"- SPECFEM3D output: {OUTPUT_DIR}")
print(f"- Project simulation: {SIMULATION_DIR}")
print("\nNext steps:")
print("1. Run DAS conversion (notebooks/02_das_conversion.ipynb)")
print("2. Preprocess the data (notebooks/03_data_preprocessing.ipynb)")
print("3. Train the model (notebooks/04_model_training.ipynb)")