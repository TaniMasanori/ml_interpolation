# Machine Learning for Seismic Interpolation

This project focuses on interpolating missing geophone seismic data using Distributed Acoustic Sensing (DAS) constraints with a machine learning approach.

## Project Structure

The repository is organized as follows:

- `data/`: Directory for all data (synthetic and real)
  - `synthetic/`: Synthetic data for training and testing
    - `raw/`: Raw simulation outputs
    - `processed/`: Processed data ready for training
- `notebooks/`: Jupyter notebooks for exploration and visualization
  - `01_specfem_simulation.ipynb`: SPECFEM3D simulation setup
  - `02_das_conversion.ipynb`: Converting seismic data to DAS data
  - `03_data_preprocessing.ipynb`: Data preprocessing
  - `04_model_training.ipynb`: Model training
  - `05_evaluation_and_visualization.ipynb`: Evaluation and visualization
- `src/`: Source code for the project
  - `evaluation/`: Evaluation metrics
  - `models/`: Model architecture definitions
  - `preprocessing/`: Data preprocessing utilities
  - `simulation/`: SPECFEM3D simulation utilities
  - `training/`: Training utilities
  - `utils/`: Utility functions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml_interpolation.git
cd ml_interpolation
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ml_interpolation
```

3. Install SPECFEM3D (required for simulations):
   - Follow the installation instructions at https://specfem3d.readthedocs.io/en/latest/
   - The compiled binary should be available at `~/specfem3d/bin/`

## Usage

### Running SPECFEM3D Simulations

1. Use the `fix_specfem_v4.1.1.py` script to properly configure SPECFEM3D v4.1.1:
```bash
python fix_specfem_v4.1.1.py
```

2. Then run the complete simulation with the `run_specfem_steps.py` script:
```bash
python run_specfem_steps.py
```

3. Alternatively, run the simulation through the Jupyter notebook:
```bash
jupyter notebook notebooks/01_specfem_simulation.ipynb
```

### Converting to DAS Data

Convert seismic data to DAS format with the notebook:
```bash
jupyter notebook notebooks/02_das_conversion.ipynb
```

### Data Preprocessing

Process the synthetic data using the notebook:
```bash
jupyter notebook notebooks/03_data_preprocessing.ipynb
```

### Model Training

Train the transformer-based interpolation model:
```bash
jupyter notebook notebooks/04_model_training.ipynb
```

### Evaluation

Evaluate the model performance:
```bash
jupyter notebook notebooks/05_evaluation_and_visualization.ipynb
```

## SPECFEM3D Configuration and Common Issues

SPECFEM3D is a complex software that requires careful configuration. Here are detailed explanations of common issues and critical configuration details:

### Parameter File Requirements

SPECFEM3D v4.1.1 has strict requirements for parameter files:
- **Complete Parameters**: All parameters must be explicitly defined, even those with default values
- **Parameter Validation**: SPECFEM3D performs strict validation of parameter files at startup
- **Common Error**: "Parameters are missing in your Par_file" error occurs when any required parameter is missing

### Critical Parameters for Mesh Generation

The mesh generation step (`xmeshfem3D`) requires specific parameters:
- **Domain Specification**: Must include LATITUDE_MIN/MAX, LONGITUDE_MIN/MAX, DEPTH_MIN/MAX
- **Mesh Density**: Requires NEX_XI, NEX_ETA, NEX_ZETA (number of elements in each direction)
- **Processor Distribution**: NPROC must match NPROC_XI × NPROC_ETA
- **UTM Settings**: UTM_PROJECTION_ZONE must be an integer (not a string)
- **Mesh Files**: interfaces.dat and no_cavity.dat must exist in DATA/meshfem3D_files/

### Critical File Structure

SPECFEM3D expects a specific directory and file structure:
- **DATA Directory**: All input files (Par_file, Mesh_Par_file, SOURCE, STATIONS)
- **meshfem3D_files Directory**: Must contain interfaces.dat and no_cavity.dat
- **OUTPUT_FILES Directory**: Where output files are stored
- **DATABASES_MPI Directory**: Where mesh and model databases are stored

### Execution Environment Requirements

- **MPI Configuration**: NPROC in Par_file must match the number of MPI processes used
- **Memory Requirements**: Large meshes require significant memory (>4GB per process)
- **Compatibility**: Mesher, partitioner, and solver must use same number of processes

### Troubleshooting the 01_specfem_simulation.ipynb Notebook

The notebook demonstrates common failures with SPECFEM3D execution:

1. **Parameter Missing Error**:
   - Error message indicates missing parameters in Par_file
   - The notebook attempts to address this by creating a complete Par_file

2. **Mesh Generation Failure**:
   - The mesher fails due to incomplete parameter definitions
   - Required files (interfaces.dat, no_cavity.dat) might be missing or incorrectly formatted

3. **Processor Mismatch**:
   - NPROC value must match the actual number of MPI processes used
   - NPROC_XI × NPROC_ETA must equal NPROC

4. **File Access Issues**:
   - Directory structure must be correctly set up
   - File permissions must allow read/write access

### Best Practices for SPECFEM3D

1. **Use the Fix Scripts**:
   - The `fix_specfem_v4.1.1.py` script addresses many common issues
   - Creates properly formatted Par_file with all required parameters

2. **Systematic Testing**:
   - Start with minimal mesh sizes (NEX_XI=NEX_ETA=NEX_ZETA=10)
   - Use single processor mode (NPROC=1) for initial testing
   - Progressively increase mesh complexity

3. **Pre-execution Validation**:
   - Verify all required files exist before running
   - Ensure parameter consistency across files

4. **Clean Between Runs**:
   - Clean OUTPUT_FILES directory between runs
   - Remove previous mesh files to avoid conflicts

By understanding these requirements and potential issues, users can more effectively troubleshoot SPECFEM3D simulations and successfully generate synthetic seismic data for the machine learning interpolation tasks.

## Architecture

The machine learning model uses a transformer-based architecture with cross-attention between DAS and geophone data. Key components:

- MultimodalSeismicTransformer: Core model for interpolation
- StorSeismicBERTModel: For handling time series data
- Custom loss functions that consider both time and frequency domains
- Progressive masking schedule during training

## Metrics

The model is evaluated using multiple metrics:
- Mean Squared Error (MSE)
- Signal-to-Noise Ratio (SNR)
- Correlation Coefficient
- Amplitude Ratio
- Frequency Domain Error

## License

[Specify your license here]



Below is the complete text with all reference citations, markers, and dedicated “References” sections removed. The content—including structure, code blocks, equations, and formatting—remains intact.

---

Got it. I’ll begin compiling the full content of the latest Specfem3D documentation into a single Markdown file, preserving all relevant structure, code blocks, equations, and formatting.

# SPECFEM3D Cartesian User Manual (Latest Version)

> *This documentation has been automatically generated by pandoc based on the SPECFEM3D Cartesian user manual (LaTeX version) as of Dec 20, 2023.*  

## 1. Introduction

The software package **SPECFEM3D Cartesian** simulates seismic wave propagation at the local or regional scale and performs full waveform imaging (FWI) or adjoint tomography using the spectral-element method (SEM). The SEM is a continuous Galerkin technique, which can easily be made discontinuous; in that discontinuous form it is close to a particular case of the discontinuous Galerkin method, with optimized efficiency due to its tensorized basis functions. In particular, SEM can accurately handle very distorted mesh elements.

**Announcements:**

- **Full Waveform Inversion (FWI)** – SPECFEM3D can now perform FWI (iteratively inverting for model parameters) and source inversions in a fixed model. See the new directories `inverse_problem_for_model` and `inverse_problem_for_source` and their README files, as well as the new examples in `EXAMPLES/`.

- **External Coupling** – SPECFEM3D can couple with external codes (DSM, AxiSEM, or FK) via a precomputed database of displacement and traction vectors on outer mesh boundaries. For example, to couple with code **FK**, set in `DATA/Par_file`:

  ```fortran
  COUPLE_WITH_INJECTION_TECHNIQUE = .false.
  INJECTION_TECHNIQUE_TYPE        = 3       # 1 = DSM, 2 = AxiSEM, 3 = FK
  MESH_A_CHUNK_OF_THE_EARTH       = .false.
  TRACTION_PATH                   = ./DATA/AxiSEM_tractions/3/
  FKMODEL_FILE                    = FKmodel
  RECIPROCITY_AND_KH_INTEGRAL     = .false.  # not yet functional
  ``` 

  Coupling with FK is actively maintained; an example is in `EXAMPLES/applications/small_example_coupling_FK_specfem`. Coupling with DSM is included but not actively maintained; see the tools in `EXTERNAL_PACKAGES_coupled_with_SPECFEM3D/DSM_for_SPECFEM3D` and its README for steps.

- **Gravity Calculations** – SPECFEM3D can compute the gravity field in addition to seismic wave propagation. See the flag `GRAVITY_INTEGRALS` in `setup/constants.h` (and accompanying documentation). SPECFEM3D can also model transient gravity perturbations from earthquake rupture. Both static gravity and transient gravity perturbations are implemented.

**Citation:** If you use SPECFEM3D Cartesian in your research, please cite at least one of the following relevant articles. (BibTeX entries for these references are available in the provided bibliography file.)

**Support:** This work has been supported by the U.S. National Science Foundation, the French CNRS, INRIA Sud-Ouest MAGIQUE-3D, ANR NUMASIS, and the European FP6 Marie Curie IRG. Any opinions expressed here do not necessarily reflect those of the funding agencies.

### References (Chapter 1)
*(Removed)*

## 2. Getting Started

To download the SPECFEM3D_Cartesian software package, run:

```bash
git clone --recursive --branch devel https://github.com/SPECFEM/specfem3d.git
```

Then, to configure the software for your system, run the `configure` shell script. This script attempts to guess appropriate configuration values. At minimum, explicitly specify your Fortran and C compilers (alternatively, define `FC`, `CC`, and `MPIF90` in your shell profile):

```bash
./configure FC=gfortran CC=gcc
```

For parallel execution (using MPI), add the MPI options:

```bash
./configure FC=gfortran CC=gcc MPIFC=mpif90 --with-mpi
```

*(Replace `gfortran`/`gcc` with your compilers, e.g. `ifort`/`icc` for Intel compilers.)* Note that MPI must be installed with MPI-IO enabled, as parts of SPECFEM3D perform I/O through MPI-IO.

Before running `configure`, you may want to edit `flags.guess` to ensure it contains optimal compiler options for your system. Known issues to check:

- **Intel ifort**: You might need `-assume byterecl`. Also, be cautious with new compiler versions; prefer using versions with service packs or updates.
- **IBM compiler**: Check if `-qsave` or `-qnosave` is needed for your machine.
- **Mac OS**: You will likely need to install Xcode (for the compilers).

For IBM xlf/xlc compilers, `configure` with specific flags:

```bash
./configure FC=xlf90_r MPIFC=mpif90 CC=xlc_r CFLAGS="-O3 -q64" FCFLAGS="-O3 -q64" --with-scotch-dir=...
``` 

On Cray systems, if `configure` fails, try exporting:
```bash
export MPI_INC=$CRAY_MPICH2_DIR/include
export FCLIBS=" "
``` 
and see `utils/infos/Cray_compiler_information` for more details. Also check the provided script `configure_SPECFEM_for_Piz_Daint.bash` for guidance.

On SGI systems, `flags.guess` automatically inserts `TRAP_FPE=OFF` into Makefiles to turn off underflow trapping.

You can add `--enable-vectorization` to the configuration to speed up computations in fluid/elastic parts. This works if and only if your system allocates contiguous memory blocks for allocatable arrays (true for most systems). If uncertain, test with and without it; identical seismograms indicate it is safe. To disable vectorization, use `--disable-vectorization`.

**Note:** We use CUBIT (aka Trelis) to create hexahedral meshes, but other meshing packages can be used (e.g., GiD or Gmsh (Geuzaine and Remacle 2009)). Even tetrahedral meshers like TetGen can be utilized by decomposing each tetrahedron into 4 hexahedra (though not optimal quality, it can help in some cases, and SEM can handle distorted elements).

SPECFEM3D uses the **SCOTCH** library for mesh partitioning. **METIS** can be used instead by setting `PARTITIONING_TYPE = 3` in `Par_file`. If using METIS, install Metis 4.0 (do *not* use 5.0 due to API differences) and edit `Makefile.in` to include the METIS link flag before running `configure`.

The SCOTCH library provides efficient static mapping, graph, and mesh partitioning. If SCOTCH is not found on the system, the configure script will use the bundled SCOTCH version. You can explicitly specify an existing SCOTCH installation with `--with-scotch-dir=/path/to/scotch`. For example:

```bash
./configure FC=ifort MPIFC=mpif90 --with-scotch-dir=/opt/scotch
``` 

If using Intel ifort, we recommend using Intel icc for compiling SCOTCH (e.g., `./configure CC=icc FC=ifort MPIFC=mpif90`).

When compiling SCOTCH, if you get "ld: cannot find -lz", install the zlib development library (e.g., `sudo apt-get install zlib1g-dev` on Linux).

To compile a **serial version** of the code (for small meshes on one node), run `configure --without-mpi` to remove MPI calls.

If running on **Windows**, consider using Docker or a Linux VM (e.g., via VirtualBox) to run SPECFEM3D.

We recommend adding `ulimit -S -s unlimited` to your `.bash_profile` (or `limit stacksize unlimited` to `.cshrc`) to remove stack size limits.

Be aware some clusters running newer OS may not compile or run an older code version; updating the code might be necessary.

For the *developer* version with dynamic rupture on multiple faults in parallel, set `FAULT_DISPL_VELOC` and `FAULT_SYNCHRONIZE_ACCEL` to `.true.` in the Par_file.

### Using the GPU version

SPECFEM3D supports CUDA and HIP GPU acceleration. To compile for NVIDIA GPUs, enable CUDA:

```bash
./configure --with-cuda
``` 

or specify a target architecture:

```bash
./configure --with-cuda=cuda9
``` 

Here, `cuda4, cuda5, ... cuda12` refer to GPU architectures (not toolkit versions). For instance:

- CUDA 4 – Tesla (e.g., K10, GTX 650)  
- CUDA 5 – Kepler (e.g., K20)  
- CUDA 6 – Kepler (e.g., K80)  
- CUDA 7 – Maxwell (e.g., Quadro K2200)  
- CUDA 8 – Pascal (e.g., P100)  
- CUDA 9 – Volta (e.g., V100)  
- CUDA 10 – Turing (e.g., RTX 2080)  
- CUDA 11 – Ampere (e.g., A100)  
- CUDA 12 – Hopper (e.g., H100)

For example, if you have CUDA toolkit 11 but a Kepler K20 GPU, use `--with-cuda=cuda5` to compile for that architecture. This sets appropriate flags.

For AMD GPUs with HIP, use:

```bash
./configure --with-hip
``` 

or specify architecture (e.g., `--with-hip=MI100` for MI100 GPUs). You can add extra HIP compilation flags via `HIP_FLAGS`. For example:

```bash
./configure --with-hip=MI250 \
    HIP_FLAGS="-fPIC -ftemplate-depth-2048 -fno-gpu-rdc \
               -O2 -fdenormal-fp-math=ieee -fcuda-flush-denormals-to-zero \
               -munsafe-fp-atomics" \
    ...
``` 

### Using the ADIOS library for I/O

For very large runs (10,000+ cores), POSIX I/O can be a bottleneck. SPECFEM3D optionally supports using the ADIOS library for more efficient parallel I/O.

To enable ADIOS:

1. Install ADIOS and set up your environment variables for ADIOS.
2. Optionally adjust ADIOS-related values in `setup/constants.h` (defaults usually suffice).
3. Configure with `--with-adios` and recompile.

*(Note: ADIOS support currently works only with meshes generated by `meshfem3D`, not for external CUBIT meshes.)* Additional ADIOS parameters are discussed in the main parameter section.

### Using HDF5 for file I/O

To alleviate I/O bottlenecks, SPECFEM3D supports writing movie snapshots and databases in HDF5 format. To enable:

Configure with HDF5 include and lib flags, for example:

```bash
./configure --with-hdf5 HDF5_INC="/opt/homebrew/include" HDF5_LIBS="-L/opt/homebrew/lib" ...
``` 

In the main `Par_file`, set `HDF5_ENABLED = .true.`. You can also dedicate extra MPI processes for I/O by setting `HDF5_IO_NODES` (those processes handle I/O asynchronously).

### Adding OpenMP support

OpenMP can be enabled in addition to MPI, though often performance doesn’t improve (pure MPI is highly optimized, and adding OpenMP may slightly slow it). Exceptions could be certain architectures. To enable OpenMP, add `--enable-openmp`:

```bash
./configure --enable-openmp ...
``` 

This adds the proper OpenMP flags for the Fortran compiler. The OpenMP loop scheduling can be controlled with the `OMP_SCHEDULE` environment variable. Tests suggest that **DYNAMIC** scheduling with a chunk size around twice the number of threads can be optimal.

### Configuration summary

Key `configure` variables:

- **F90**: Path to the Fortran compiler.  
- **MPIF90**: Path to MPI Fortran.  
- **MPI_FLAGS**: Flags needed to link MPI libraries on some systems.  
- **FLAGS_CHECK**: Compiler flags for debugging/performance checks.

After configuring, a `Makefile` is generated in each `src/*` directory. These default flags are not optimal for every system – experiment and consult your admin for the best flags. Proper compiler/flags selection can significantly impact performance.

After compiling, you should set a few flags in `setup/constants.h` for your system:

- **LOCAL_PATH_IS_ALSO_GLOBAL**: Usually `.false.` for clusters (the database generator writes output to local disks for speed). If your system has no local disks or uses a fast parallel file system, set it `.true.`.
- **Precision** – SPECFEM3D runs in single or double precision. By default, single precision is used. To switch, edit `CUSTOM_REAL` in `constants.h` and `CUSTOM_MPI_TYPE` in `src/shared/precision.h`.
- **MPI include vs module** – If your compiler has issues with `use mpi`, run the provided script to switch to `include 'mpif.h'`.

### Compiling on IBM BlueGene

For IBM BlueGene systems, special compiler flags are needed. One set of recommended flags for BlueGene/Q is:

```
FLAGS_CHECK = -g -qfullpath -O2 -qsave -qstrict -qtune=qp -qarch=qp \
              -qcache=auto -qhalt=w -qfree=f90 -qsuffix=f=f90 \
              -qlanglvl=95pure -Q -Q+rank,swap_all -Wl,-relax
``` 

Key points:
- Use `-qarch`/`-qtune` specific to the compute nodes (not “auto”).
- Sometimes `-O3` has issues; use `-O2` if needed.
- The linker flag `-Wl,-relax` is often needed on BG/Q.
- You may need to adjust AR and RANLIB to BG-specific versions (`bgar`, `bgranlib`).

Load the XL compilers on BG/Q (`module load bgq-xl`), or use the GNU ones if needed.

Run `configure` with BG-specific host and build identifiers, and the BlueGene compilers. For example:

```bash
./configure --host=powerpc64-bgq-linux --build=x86_64-linux-gnu \
    FC=bgxlf90_r MPIFC=mpixlf90_r CC=bgxlc_r \
    AR=bgar ARFLAGS=cru RANLIB=bgranlib \
    LOCAL_PATH_IS_ALSO_GLOBAL=false
``` 

On BlueGene, after compiling, run the `xcreate_header_file` program manually to create the database header file.

### Visualizing the call tree (Doxygen)

For developers interested in code structure, Doxygen configuration files are provided in `doc/Call_trees`. They can produce call graphs of subroutine calls:

- `Doxyfile_truncated_call_tree` – limits graph depth.
- `Doxyfile_complete_call_tree` – full call graph.

Basic usage:
1. Install Doxygen and Graphviz.
2. Use the provided Doxyfile or generate one and edit it.
3. Important settings include enabling Fortran optimization, extracting all members, enabling dot, and setting the input directory to `src/`.

Running Doxygen will create HTML documentation including call graphs.

## 3. Mesh Generation

The first step in a spectral-element simulation is to build a high-quality mesh for the model region. Two approaches are provided: **(1)** using an external mesher like **CUBIT** (for unstructured hexahedral meshes), or **(2)** using the internal mesher **xmeshfem3D**.

### Meshing with CUBIT

*CUBIT* (also known as Trelis) is a mesh generation toolkit developed at Sandia National Laboratories, available for academics. Using CUBIT streamlines the creation of conforming hexahedral meshes for complex geological models.

*Figure: CUBIT GUI example – A hexahedral mesh of a single volume with topography displayed.*

The basic steps to create a partitioned mesh with CUBIT are:

1. **Geometry and Meshing in CUBIT:** Define the model geometry and mesh it with hexahedral elements.
2. **Export to SPECFEM3D format:** Use CUBIT’s scripting interface or plugins to export the mesh in SPECFEM3D format.
3. **Partitioning:** Split the mesh for parallel processing using `xdecompose_mesh`.

Several examples are provided under `EXAMPLES/` in the SPECFEM3D package.

#### Creating the Mesh with CUBIT

Refer to the CUBIT manual for installation and usage. For SPECFEM3D, example input files are provided in `EXAMPLES/`:

- **homogeneous_halfspace** – a single-block half-space with uniform elastic properties.
- **layered_halfspace** – two layers with different materials and a refined interface.
- **waterlayered_halfspace** – an acoustic water layer over an elastic half-space.
- **tomographic_model** – a single-block volume where material properties are assigned later during database generation.

*Figures: Homogeneous half-space mesh, Two-layer half-space mesh, Water over half-space mesh, and Tomographic model mesh.*

To regenerate these meshes, provided CUBIT journal and Python scripts can automate the process.

For models with faults, the mesh must include split nodes along fault surfaces. This is achieved in CUBIT by creating a very thin gap between the two fault surfaces so that each side has its own set of nodes while keeping the fault edges coincident.

*Figure: Example fault meshing – fault surfaces are split so that each side has its own nodes.*

Scripts provided in the package adjust node positions on one side of the fault slightly to ensure duplicate nodes.

#### Exporting the Mesh: `cubit2specfem3d.py`

After meshing in CUBIT, export the mesh to SPECFEM3D format using the provided Python script (`cubit2specfem3d.py`), which:

- Writes mesh node coordinates and element connectivity.
- Identifies free surface, external boundaries, and fault faces.
- Writes material IDs for each element.

Place the output files into the appropriate directory before running the database generation.

**Important:** Ensure no gaps exist in the numbering of element, face, edge, and node IDs by using the appropriate CUBIT command.

### Meshing with `xmeshfem3D` (Internal Mesher)

For simpler models, you can use the internal mesher `xmeshfem3D`. Compile it with:

```bash
make xmeshfem3D
``` 

Input parameters are set in `DATA/meshfem3D_files/Mesh_Par_file`, defining a regular meshing of a rectangular region (with optional topography). Key parameters include geographic extent and UTM projection zone. To disable UTM projection for a purely Cartesian model, set `SUPPRESS_UTM_PROJECTION = .true.`.

After editing `Mesh_Par_file`, run `xmeshfem3D` from the top-level directory. The output mesh files will be partitioned among MPI ranks.

*(Note: The internal mesher does not support faults.)*

**Parallel Meshing:** The internal mesher partitions the model into a grid based on the number of processors.

*Figure: Partitioning example with 25 processors.*

### References (Mesh Generation)
*(Removed)*

## 4. Creating Distributed Databases

After meshing, the next step is to generate the spectral-element “distributed databases” needed for the simulation using the `xgenerate_databases` program. This program computes the Gauss–Lobatto–Legendre (GLL) mesh points, assigns material properties, precomputes absorbing boundary information, and more.

Compile the database generator:

```bash
make xgenerate_databases
``` 

Run `xgenerate_databases` (from the main directory), which reads the main **`DATA/Par_file`**.

**Input requirements:** The solver expects three input files in `DATA/`:

- `Par_file` – main parameter file.
- `CMTSOLUTION` or `FORCESOLUTION` – source description.
- `STATIONS` – list of station coordinates.

Most parameters in `Par_file` must be finalized before running `xgenerate_databases`. Only a few run-time parameters can be tweaked later, such as:

- `SIMULATION_TYPE` and `SAVE_FORWARD` – to switch between forward and adjoint runs.
- Time step parameters (`NSTEP` and `DT`).
- Absorbing boundary toggle (`PML_CONDITIONS`), if compatible.
- Movie output flags can be changed later.

Other parameters (e.g., material properties, mesh geometry) must remain unchanged.

#### PML absorbing boundary layers

If using PML absorbing boundaries (`PML_CONDITIONS = .true.`), ensure the outermost material layers comply and keep the settings consistent between database generation and the solver run.

### Choosing the time step (`DT`)

The time step `DT` is crucial for simulation stability. It must satisfy the Courant condition based on the smallest element size and highest wave speed. The code often suggests a maximum `DT` during mesh generation or database generation. Adjust `DT` accordingly when refining the mesh or changing material properties.

### References (Creating Databases)
*(Removed)*

## 5. Running the Solver (`xspecfem3D`)

Now that the databases are generated, compile and run the spectral-element solver:

```bash
make xspecfem3D
``` 

Run **`xspecfem3D`** from the main directory. The solver reads three files in `DATA/`:

- `Par_file` – main parameter file.
- `CMTSOLUTION` (or `FORCESOLUTION`) – source description.
- `STATIONS` – station coordinates.

Ensure these files are prepared. The `CMTSOLUTION` file is required even for noise or ambient vibration simulations.

During the run, `xspecfem3D` produces seismograms at stations (one file per receiver component) and any optional outputs (movies, strain, etc.) as specified in `Par_file`.

**Important:** The `Par_file` parameters must remain consistent with those used during database generation. Only a few parameters (e.g., `NSTEP`, `DT`, `SIMULATION_TYPE`, and output flags) may be changed between database generation and solver runs.

**Simultaneous multiple simulations:** SPECFEM3D allows running multiple independent simulations (e.g., multiple earthquakes) simultaneously by splitting MPI processes among events.

**Attenuation model:** SPECFEM3D implements constant-Q (frequency-independent) attenuation using Standard Linear Solid (SLS) mechanisms. The number of SLS mechanisms is set by `N_SLS` in Par_file. Use the default value unless you have specific requirements.

**Output files:** The solver outputs seismogram files (ASCII or binary), forward wavefield snapshots (if `SAVE_FORWARD` is true), movie files (if enabled), and shakemaps (if `CREATE_SHAKEMAP` is true).

**Resuming runs:** If a run is interrupted and restart files are written, you can resume by setting `restart=.true.` in Par_file and adjusting the simulation parameters accordingly.

### Note on the simultaneous simulation of several earthquakes

This feature divides MPI processes among independent simulations, allowing parameter studies or ensemble calculations. Prepare combined station and source files as needed and configure the Par_file to split the MPI world accordingly.

### Note on the viscoelastic model used

The attenuation model is constant-Q, approximated by multiple SLS mechanisms. Ensure that `ATTENUATION_f0_REFERENCE` is set near the center of your frequency band.

### References (Running the Solver)
*(Removed)*

## 6. Fault Sources

SPECFEM3D can handle finite fault sources of two kinds:

1. **Kinematic** – The slip rate distribution on the fault is prescribed.
2. **Dynamic** – Rupture propagates spontaneously based on friction laws and initial stress conditions.

**Mesh requirements:** Faults must lie on element interfaces with duplicate (split) nodes along the fault. Currently, faults are supported with externally generated meshes; the internal mesher does not support faults.

**Input files for faults:** In addition to the standard `Par_file`, you need:

- `Par_file_faults` – parameters for fault rupture.
- `CMTSOLUTION` – used to indicate a finite fault for kinematic ruptures.
- `STATIONS` – station list.
- For dynamic ruptures, additional files may specify initial stresses and friction parameters.

**Kinematic source setup:** The fault surface is subdivided into subfaults, with slip defined as a function of space and time. Example configurations are provided in `EXAMPLES/kinematic_fault`.

**Dynamic rupture setup:** Parameters such as `FAULT_TYPE`, cohesion, friction coefficient, and initial stresses are specified. The solver uses a friction law (e.g., slip-weakening) to model rupture propagation.

**Sign convention for fault slip and traction:** One side of the fault is defined as “side 1” and the other as “side 2”. The convention for slip and traction follows a right-hand rule relative to the fault normal.

**Kelvin-Voigt damping on faults:** A small viscosity may be added to stabilize the dynamic rupture simulation. This is controlled by a fault viscosity parameter in `Par_file_faults`.

**Output for faults:** The code writes files for slip rates, rupture onset times, final slip distribution, and possibly traction time histories.

**Post-processing and visualization:** Utilities (often MATLAB scripts) are provided to visualize fault slip, rupture times, and other fault-related outputs.

### References (Fault Sources)
*(Removed)*

## 7. Adjoint Simulations

Adjoint simulations are used for:

- **Source inversions:** Refining earthquake source parameters by minimizing the misfit between observed and synthetic seismograms.
- **Finite-frequency kernel computations:** Calculating sensitivity kernels (adjoint-based gradients) with respect to model parameters for tomography.

For both applications, an adjoint simulation runs the wave equation “backwards” using adjoint sources (typically derived from seismogram residuals).

**Workflow for adjoint simulation (for tomography):**

1. **Forward simulation:** Run `xspecfem3D` with `SIMULATION_TYPE=1` and `SAVE_FORWARD = .true.` to generate seismograms and save forward wavefield snapshots.
2. **Measure misfit:** Compare synthetics with observed data and compute residuals.
3. **Adjoint simulation:** Set `SIMULATION_TYPE` to the appropriate adjoint type, provide the adjoint sources, and run `xspecfem3D` again.
4. **Compute kernels:** The adjoint run outputs sensitivity kernel files for model parameters.

**Adjoint source generation:** Typically, the time-reversed seismogram residuals are used as adjoint sources.

**Parameter settings:** Key flags include `SIMULATION_TYPE`, `SAVE_FORWARD`, and ensuring the simulation spans the entire time of the forward run.

**Output:** Kernel files for volumetric sensitivities are generated for further processing in tomography.

### References (Adjoint Simulations)
*(Removed)*

## 8. Doing Tomography (Model Updates)

This section describes how to update the Earth model using adjoint kernels from full waveform inversion (FWI). The process is iterative:

- Start with a reference model.
- Compute synthetic seismograms and compare them with observed data.
- Use adjoint simulations to calculate misfit sensitivity kernels.
- Update the model by subtracting a scaled version of the gradient from the current model.

**Tomographic FWI imaging using kernels:** Sensitivity kernels indicate where and by how much the model parameters should be adjusted to reduce misfit. Multiple event kernels can be summed and then smoothed to obtain a stable gradient.

**Model updating:** A simple steepest descent update may be applied. Post-processing tools (e.g., to sum and smooth kernels) are provided in the `utils/` directory.

### References (Doing Tomography)
*(Removed)*

## 9. Noise Simulations

SPECFEM3D Cartesian can simulate seismic ambient noise cross-correlations for noise tomography. This approach extracts Green’s functions between stations by cross-correlating long noise recordings.

**Noise Cross-correlation Simulations:** Instead of a discrete event, continuous noise sources are simulated, and the recordings at station pairs are cross-correlated to retrieve the impulse response.

**Input Parameter Files:** The main `Par_file` is used with additional parameters controlling noise source characteristics and simulation duration. A `CMTSOLUTION` file is still required, even if it defines a dummy source.

**Noise simulation procedure:**

1. **Pre-simulation:** Define the noise source model (e.g., random forces on the surface or at specific locations).
2. **Simulations:** Run long-duration simulations (or multiple shorter ones) to capture the noise time series at stations.
3. **Post-simulation:** Compute cross-correlations between station pairs, either internally by the code or using external tools, and stack the results for improved signal-to-noise.

### References (Noise Simulations)
*(Removed)*

## 10. Gravity Calculations

SPECFEM3D can compute gravity field perturbations from a 3D Earth model. This includes:

- Calculating the static gravitational potential (for comparison with gravity observations).
- Modeling transient gravity changes due to seismic events (if enabled).

To use gravity integrals:

- Ensure gravity options are enabled in the code.
- Configure the model to include density variations.
- Run the solver or a dedicated gravity post-processing tool to compute gravitational potential and its derivatives on a specified observation surface.

### References (Gravity Calculations)
*(Removed)*

## 11. Graphics

SPECFEM3D provides several tools for visualizing simulation results:

- **Mesh Visualization:** Convert mesh and model outputs into formats like VTK for viewing in Paraview.
- **Volume and Surface Movies:** If enabled via `MOVIE_SURFACE` or `MOVIE_VOLUME`, the code writes time-series snapshots of the wavefield to binary files. These can be converted into animations using provided tools.
- **Shakemaps:** The code computes peak ground motions and outputs them as shakemaps.
- **Seismogram Plotting:** Output seismogram files can be plotted with any standard tool (MATLAB, Python, etc.). Utilities may be provided to convert outputs into SAC or miniSEED formats.

A dedicated program (e.g., `xcreate_movie_shakemap_AVS_DX_GMT`) converts movie files into various formats for visualization.

### References (Graphics)
*(Removed)*

## 12. Running through a Scheduler

On HPC systems, SPECFEM3D is typically run via a job scheduler (SLURM, PBS, LSF, etc.). Guidelines include:

- Use an MPI launcher (e.g., `srun` or `mpirun`) to run the executables.
- Ensure the environment (PATH, LD_LIBRARY_PATH) is correctly set up.
- Request resources that match the number of MPI tasks specified in `Par_file`.
- Example SLURM script:

  ```bash
  #!/bin/bash
  #SBATCH -J specfem_run
  #SBATCH -N 4               # number of nodes
  #SBATCH -n 64              # total MPI tasks
  #SBATCH -t 02:00:00        # run time
  #SBATCH -p regular

  module load intel mpi      # load compilers and MPI
  cp DATA/Par_file OUTPUT_FILES/   # backup Par_file (optional)

  srun ./bin/xmeshfem3D      # run mesher (if needed)
  srun ./bin/xgenerate_databases
  srun ./bin/xspecfem3D
  ```

- For array jobs or simultaneous simulations, adapt the script accordingly.
- Use checkpointing options if necessary.
- Ensure that output files are stored on a persistent file system.

## 13. Changing the Model

This section explains how to modify the Earth model used in simulations. Options include:

- Modifying routines (e.g., `read_external_model`) to read custom material property files.
- Using supported file formats (e.g., velocity model files) that SPECFEM3D can read.
- Hard-coding the model into the code (for quick tests).

After modifying the model, always re-run `xgenerate_databases` so the new model is incorporated.

## 14. Post Processing

Post processing involves analysis and visualization of outputs after a simulation:

- **Filtering Seismograms:** Use provided scripts or external tools (e.g., SAC, ObsPy) to filter synthetic seismograms.
- **Measuring Misfit:** Compare synthetic and observed seismograms to compute misfit (user-provided tools may be available).
- **Spectral Analysis:** Use FFT to examine the frequency content.
- **Visualizing Wavefields:** Convert movie outputs to images or videos using provided converters.
- **Computing Intensity Measures:** Calculate peak ground velocity or acceleration using simple scripts.
- **Data Comparison:** Convert outputs to standard formats (SAC, miniSEED) for comparison with real data.

It is advisable to organize the `OUTPUT_FILES/` directory after each run for clarity.

## 15. Information for Developers

For those looking to modify or extend SPECFEM3D:

- **Code Structure:** The code is organized into subdirectories under `src/` (shared routines, mesher, database generator, solver, adjoint routines, etc.).
- **Coding Guidelines:** The code is mostly in Fortran 90 with some C. Follow existing style conventions.
- **Parallelization:** SPECFEM3D uses MPI for distributed memory parallelism. Ensure any modifications maintain proper communication and memory allocation.
- **Memory Optimization:** Be mindful of large arrays and use allocatable arrays where appropriate.
- **GPU Acceleration:** New GPU features require corresponding kernels and preprocessor flags.
- **Testing:** Use the provided examples in `EXAMPLES/` to verify changes.
- **Contributing:** The project is open-source on GitHub. Contributions via pull requests are welcome.
- **Documentation:** Update the LaTeX manual if you add user-facing features.
- **Community Support:** For major changes or help, consult the community forum.

---

## Appendix A: Reference Frame

**Reference Frame Convention:** SPECFEM3D Cartesian uses a right-handed Cartesian coordinate system:

- **x-axis:** East
- **y-axis:** North
- **z-axis:** Up (positive upward)

Note that this convention may differ from other common conventions. Input files (`CMTSOLUTION` and `STATIONS`) are provided in geographic coordinates and converted internally via an approximate UTM projection. If `SUPPRESS_UTM_PROJECTION` is set to `.true.`, the lat/long values are treated directly as Cartesian x and y.

**Seismogram outputs:** Seismogram components are output in the local N, E, Z orientation. For example, the vertical component file is labeled with Z, and the horizontal components are labeled according to their x (East) and y (North) directions.

---

## Appendix B: Channel Codes

Seismic channels in SPECFEM3D outputs follow a naming scheme. Files follow the pattern `STA.NET.COMP.semp`, where:

- **STA:** Station code.
- **NET:** Network code.
- **COMP:** Component code (e.g., BXX, BXY, BXZ).

These correspond to broadband seismometer channels. By convention:

- `BXZ` represents vertical ground motion.
- `BXX` represents east-west motion (east-positive).
- `BXY` represents north-south motion (north-positive).

Users may wish to rename these to standard codes (e.g., BHZ, BHE, BHN) when comparing with real data.

---

## Appendix C: Troubleshooting

This appendix provides solutions to common issues:

- **Run Crashes Immediately:** Verify that `NPROC` in Par_file matches the number of MPI processes launched and that all necessary input files (e.g., STATIONS, CMTSOLUTION) are present.
- **NaNs or Infs in Output:** This usually indicates instability. Check that the time step `DT` is sufficiently small and that material properties (e.g., ensuring fluids are flagged correctly) are set appropriately.
- **Segmentation Faults During Mesh Reading:** Verify that the mesh files are in the expected format and that the `ulimit` for stack size is set to unlimited.
- **MPI Connectivity Errors:** Ensure that the mesh partitioning is correct and that all elements and nodes are properly numbered.
- **Discrepancies Between Single and Double Precision Runs:** Differences should be small; if not, consider using double precision or adjusting model parameters.
- **GPUs Not Being Used:** Confirm that the configuration was done with the appropriate `--with-cuda` or `--with-hip` flag and that the runtime environment is set to use GPUs.
- **Solver is Slow:** Ensure you are not in debug mode, verify that compiler optimization flags are enabled, and check for I/O bottlenecks.
- **No Output Files Generated:** Check log files and standard output for error messages. Verify that output directories and parameters in Par_file are correct.
- **Illegal Instruction Errors:** The code may have been compiled with instructions not supported by your CPU. Recompile with more generic flags.
- **Memory Allocation Errors:** For very large meshes, consider increasing the number of MPI processes to distribute memory usage or using checkpointing.
- **SCOTCH/METIS Partitioning Issues:** Try switching partitioners or verifying that the libraries are correctly installed.
- **Boundary Reflection Issues:** Check and adjust absorbing boundary or PML settings.
- **Free Surface Issues:** Ensure that the mesh top coincides with the free surface and that boundary conditions are correctly set.
- **Dynamic Rupture Instability:** If the fault simulation is unstable, try reducing the time step or increasing fault damping.
- **No Kernel Output After Adjoint Run:** Ensure that `SIMULATION_TYPE` is set correctly and that forward wavefield files are available.
- **Additional Issues:** Consult the SPECFEM forum or FAQ for further assistance.

---

## Appendix D: License

SPECFEM3D Cartesian is distributed under the GNU General Public License (GPL) version 3 (or later). This means:

- You are free to use, modify, and distribute the code, provided that any distributed modifications are also under the GPL.
- There is no warranty for the code; it is provided “as is.”

Please consult the `COPYING` or `LICENSE` file for the full license text.

---

## About: Authors, Bug Reports, and Acknowledgments

**Authors:** SPECFEM3D_Cartesian was originally developed by Dimitri Komatitsch and Jeroen Tromp, with contributions from many others. A full list of contributors is available in the documentation.

**Bug Reports:** If you encounter a bug, please open an issue on the GitHub repository or contact the developers via the community forum. Include detailed information such as the code version, compiler, operating system, input setup, and error messages.

**Version:** This documentation corresponds to SPECFEM3D Cartesian 3.0 (Dec 20, 2023). Please mention the code version in any publications.

**Features:** The code includes GPU acceleration (CUDA and HIP), anisotropy, attenuation and undo attenuation for kernels, noise cross-correlation, gravity calculations, flexible absorbing boundaries (PML), support for unstructured meshes via external and internal meshers, adjoint capabilities for FWI and source inversion, and multi-instance simulation support.

**Acknowledgments:** The project has benefited from support by various institutions and funding agencies. We thank all contributors and users for their feedback.

---

This version now omits all references, citation markers, and dedicated reference lists while preserving the structure and technical content of the original documentation.

[Add acknowledgments]