# ML Interpolation Project Workflow

## 1. SPECFEM Model and Setting Files Generation

### Core Components
- **Parameter Management**: Configuration of SPECFEM simulation parameters
- **Velocity Model Creation**: Generation of layered velocity models
- **Source & Receiver Setup**: Definition of seismic sources and receiver arrays
- **Configuration Files**: Automatic generation of Par_file, Mesh_Par_file, SOURCE, STATIONS

### Key Scripts
- `param_manager.py`: Central tool for managing all SPECFEM parameters
- `create_simulation_configs.py`: Generate simulation configurations
- `parameter_sets/*.json`: Predefined parameter templates

### Workflow
1. Load parameter templates
2. Customize model geometry, material properties, simulation settings
3. Generate configuration files
4. Create interface files for layered models

## 2. Model and Source/Receiver Visualization

### Core Components
- **Velocity Model Plots**: 2D/3D visualizations of Vp, Vs, density models
- **Source Visualization**: Plot locations and characteristics of seismic sources
- **Receiver Layout**: Visualize station/fiber distributions and orientations
- **Geological Layer Display**: Show interfaces between layers

### Key Scripts
- `plot_velocity_model.py`: Advanced velocity model visualization
- `plot_simple_velocity_model.py`: Simplified model visualization

### Visualization Types
- Cross-sectional plots of material properties
- 3D model representations
- Source-receiver geometry visualization
- Interactive parameter adjustment

## 3. SPECFEM Meshing and Simulation

### Core Components
- **Mesh Generation**: Creation of finite element meshes
- **Database Generation**: Preparation of material properties databases
- **Wave Propagation**: Simulation of seismic wave propagation
- **Output Collection**: Gathering and organizing simulation results

### Key Scripts
- `src/simulation/specfem_runner.py`: Core simulation execution engine
- `run_specfem.py`: High-level simulation runner with logging

### Simulation Steps
1. Create simulation directory structure
2. Generate mesh with xmeshfem3D
3. Create databases with xgenerate_databases  
4. Run wave propagation with xspecfem3D
5. Collect seismograms and wavefield snapshots

## 4. DAS Data (Strain Rate) Conversion

### Core Components
- **Strain Calculation**: Computation of spatial derivatives along fiber
- **Gauge Length Simulation**: DAS gauge length effects implementation
- **Multi-component Integration**: Handling of directional sensitivity

### Key Scripts
- `src/simulation/das_converter.py`: Core DAS conversion library
- `convert_seismo_to_das.py`: Conversion script for seismogram data

### Conversion Process
1. Load geophone velocity data (X,Y,Z components)
2. Project onto fiber direction
3. Calculate spatial derivatives along fiber
4. Apply gauge length averaging
5. Output DAS strain rate data

## 5. Data Visualization

### Core Components
- **Shot Gathers**: Time-distance visualization of seismic data
- **Wiggle Plots**: Traditional seismic visualization format
- **Waterfall Displays**: Sequential visualization of traces
- **Comparative Plotting**: Side-by-side geophone/DAS visualization

### Key Scripts
- `plot_shot_gather.py`: Versatile shot gather visualization
- `plot_wavefield_snapshots.py`: Visualization of wave propagation

### Visualization Capabilities
- Component-wise display (X, Y, Z, strain)
- Amplitude normalization
- Time-distance color mappings
- Multi-panel comparative views

## 6. Data Preparation and Model Training

### Core Components
- **Dataset Creation**: Integration and preparation of geophone/DAS data
- **Masking Techniques**: Methods to simulate missing data
- **Model Architecture**: Transformer-based models for interpolation
- **Training Pipeline**: Efficient training with validation

### Key Scripts
- `src/preprocessing/dataset.py`: Dataset preparation classes
- `src/models/transformer.py`: ML model implementations
- `src/training/trainer.py`: Training workflows
- `src/evaluation/metrics.py`: Performance metrics

### Training Workflow
1. Prepare datasets with masking patterns
2. Configure model architecture
3. Train with cross-validation and early stopping
4. Evaluate performance with multiple metrics
5. Visualize interpolation results

## Jupyter Notebooks for Interactive Development

A Jupyter notebook for model design and visualization will include:

1. **Parameter Configuration**
   - Interactive adjustment of model dimensions and properties
   - Source frequency and mechanism configuration
   - Receiver array layout design
   
2. **Model Visualization**
   - Cross-sections of velocity and density models
   - 3D visualization of the complete model
   - Layer interface plots
   
3. **Source-Receiver Visualization**
   - Plot source locations with characteristics
   - Display receiver arrays with orientation
   - Show DAS fiber layout
   
4. **Geometry Validation**
   - Check source-receiver offsets
   - Verify model boundaries
   - Confirm proper layer definitions

5. **Parameter Export**
   - Save configured parameters to JSON
   - Generate SPECFEM input files
   - Create setup for batch processing

## Streamlined Workflow

The recommended streamlined workflow eliminates duplicated functionality:

1. **Setup**: Use `param_manager.py` as the single source for all parameter configuration
2. **Visualization**: Interactive Jupyter notebook for model design and inspection
3. **Simulation**: Consolidated `specfem_runner.py` for all simulation steps
4. **Conversion**: Integrated DAS conversion with simulation pipeline
5. **Training**: End-to-end training workflow with proper validation
6. **Evaluation**: Comprehensive metrics and visualization tools

This workflow provides a clear path from model design to final evaluation while eliminating redundant code and providing visual feedback at critical steps.