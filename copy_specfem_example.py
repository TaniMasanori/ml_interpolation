#!/usr/bin/env python3
"""
This script copies a working SPECFEM3D example and adapts it for our use case.
It directly uses a known working example to ensure compatibility.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np

# Define paths
SPECFEM_DIR = os.path.expanduser("~/specfem3d")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/synthetic/raw/simulation1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select a working example from SPECFEM3D
EXAMPLE_DIR = os.path.join(SPECFEM_DIR, "EXAMPLES/applications/homogeneous_poroelastic")
EXAMPLE_DATA_DIR = os.path.join(EXAMPLE_DIR, "DATA")

# Check if the example exists
if not os.path.exists(EXAMPLE_DATA_DIR):
    print(f"Error: Example directory {EXAMPLE_DATA_DIR} not found!")
    sys.exit(1)

print(f"=== SPECFEM3D Example Runner ===")
print(f"SPECFEM3D directory: {SPECFEM_DIR}")
print(f"Using example from: {EXAMPLE_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Clean SPECFEM directories
def clean_specfem():
    """Clean SPECFEM3D directories for a fresh start."""
    print("\nCleaning SPECFEM3D directories...")
    
    # Clean OUTPUT_FILES directory
    output_dir = os.path.join(SPECFEM_DIR, "OUTPUT_FILES")
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path) and item != "DATABASES_MPI":
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error cleaning {item_path}: {e}")
    else:
        os.makedirs(output_dir)
    
    # Ensure DATABASES_MPI exists
    databases_dir = os.path.join(output_dir, "DATABASES_MPI")
    os.makedirs(databases_dir, exist_ok=True)
    
    # Clean DATA directory
    data_dir = os.path.join(SPECFEM_DIR, "DATA")
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if item != "meshfem3D_files" and not item.startswith("."):
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error cleaning {item_path}: {e}")
    
    # Ensure meshfem3D_files directory exists
    mesh_dir = os.path.join(data_dir, "meshfem3D_files")
    os.makedirs(mesh_dir, exist_ok=True)
    
    print("SPECFEM3D directories cleaned.")

# Copy example files to SPECFEM3D DATA directory
def copy_example_files():
    """Copy files from the example to SPECFEM3D DATA directory."""
    print("\nCopying example files...")
    
    # Get source and destination directories
    src_dir = EXAMPLE_DATA_DIR
    dst_dir = os.path.join(SPECFEM_DIR, "DATA")
    
    # Copy all files from example DATA directory
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied file: {item}")
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            print(f"  Copied directory: {item}")
    
    print("Example files copied to SPECFEM3D DATA directory.")

# Create custom STATIONS file
def create_custom_stations():
    """Create a custom STATIONS file with evenly spaced stations."""
    print("\nCreating custom STATIONS file...")
    
    stations_file = os.path.join(SPECFEM_DIR, "DATA", "STATIONS")
    
    # Parameters for station layout (linear array)
    n_stations = 100
    start_x = 500.0
    end_x = 1500.0
    y_position = 1000.0
    depth = 0.0
    
    # Generate station locations
    x_positions = np.linspace(start_x, end_x, n_stations)
    
    # Write the STATIONS file
    with open(stations_file, 'w') as f:
        for i, x in enumerate(x_positions):
            # Format: STA NET LAT LON ELEVATION BURIAL
            f.write(f"ST{i+1:03d} GE {x} {y_position} 0.0 {depth}\n")
    
    # Copy to output directory
    output_stations_file = os.path.join(OUTPUT_DIR, "STATIONS")
    shutil.copy(stations_file, output_stations_file)
    
    print(f"Created custom STATIONS file with {n_stations} stations.")

# Run SPECFEM3D
def run_specfem():
    """Run SPECFEM3D mesher, partitioner, and solver."""
    print("\nRunning SPECFEM3D simulation...")
    
    # Save current directory
    current_dir = os.getcwd()
    
    # Get NPROC from Par_file
    nproc = 4  # Default value
    par_file_path = os.path.join(SPECFEM_DIR, "DATA", "Par_file")
    with open(par_file_path, 'r') as f:
        for line in f:
            if "NPROC" in line and "=" in line and not line.strip().startswith("#"):
                try:
                    nproc = int(line.split("=")[1].strip())
                    break
                except:
                    pass
    
    print(f"Using NPROC = {nproc}")
    
    # Change to SPECFEM directory
    os.chdir(SPECFEM_DIR)
    
    try:
        # Step 1: Run the mesher
        print("\n=== Step 1: Running mesher (xmeshfem3D) ===")
        cmd = f"mpirun -np {nproc} ./bin/xmeshfem3D"
        print(f"Executing: {cmd}")
        
        result = subprocess.run(cmd, shell=True, check=False, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print("⚠️ Meshing failed with error:")
            print(result.stderr)
            os.chdir(current_dir)
            return False
        
        print("✅ Meshing completed successfully")
        
        # Step 2: Run the partitioner
        print("\n=== Step 2: Running partitioner (xdecompose_mesh) ===")
        cmd = f"mpirun -np {nproc} ./bin/xdecompose_mesh"
        print(f"Executing: {cmd}")
        
        result = subprocess.run(cmd, shell=True, check=False, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print("⚠️ Partitioning failed with error:")
            print(result.stderr)
            os.chdir(current_dir)
            return False
        
        print("✅ Partitioning completed successfully")
        
        # Step 3: Run the solver
        print("\n=== Step 3: Running solver (xspecfem3D) ===")
        cmd = f"mpirun -np {nproc} ./bin/xspecfem3D"
        print(f"Executing: {cmd}")
        
        result = subprocess.run(cmd, shell=True, check=False, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print("⚠️ Solver failed with error:")
            print(result.stderr)
            os.chdir(current_dir)
            return False
        
        print("✅ Solver completed successfully")
        
        # Copy output files
        output_files_dir = os.path.join(SPECFEM_DIR, "OUTPUT_FILES")
        seismo_count = 0
        
        for file in os.listdir(output_files_dir):
            if file.endswith((".semd", ".semp", ".sema")) or file.startswith(("timestamp", "output_")):
                src_file = os.path.join(output_files_dir, file)
                dst_file = os.path.join(OUTPUT_DIR, file)
                shutil.copy(src_file, dst_file)
                seismo_count += 1
        
        print(f"\nCopied {seismo_count} output files to {OUTPUT_DIR}")
        return True
    
    finally:
        # Return to original directory
        os.chdir(current_dir)

# Main execution
def main():
    # Check that SPECFEM3D is installed
    if not os.path.exists(os.path.join(SPECFEM_DIR, "bin", "xmeshfem3D")):
        print("Error: SPECFEM3D executable not found. Make sure it's installed and compiled.")
        sys.exit(1)
    
    # Clean SPECFEM3D directories
    clean_specfem()
    
    # Copy example files
    copy_example_files()
    
    # Create custom STATIONS file
    create_custom_stations()
    
    # Run SPECFEM3D
    success = run_specfem()
    
    if success:
        print("\n✅ SPECFEM3D simulation completed successfully!")
        
        # Verify output files were created
        seismogram_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".semd")]
        if seismogram_files:
            print(f"Generated {len(seismogram_files)} seismogram files.")
        else:
            print("Warning: No seismogram files were generated.")
    else:
        print("\n❌ SPECFEM3D simulation failed.")
    
    return success

if __name__ == "__main__":
    main()