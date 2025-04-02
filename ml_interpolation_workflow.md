# ML Interpolation Workflow Guide

## 1. SPECFEM Model and Setting Files Generation

The first step involves setting up the simulation model and parameters for seismic wave propagation.

### Key Components
- Create velocity model (Vp, Vs, density) with layered structures
- Define source parameters (location, frequency, mechanism)
- Set up receiver arrays (geophones, DAS fiber) 
- Generate configuration files for SPECFEM3D

### Files Used
- `param_manager.py`: Main tool for parameter management
- `parameter_sets/*.json`: Parameter templates
- Model design Jupyter notebook: `notebooks/01_model_design_visualization.ipynb`

### Process
1. Start with the Jupyter notebook to interactively design models
2. Modify material properties and layer interfaces
3. Configure source and receiver geometry
4. Export SPECFEM input files (Par_file, SOURCE, STATIONS, interfaces)

## 2. Model and Source/Receiver Visualization

Visualize the model and experimental setup before running simulations.

### Key Components
- 2D/3D visualization of velocity models
- Source-receiver geometry plotting
- Layer interface visualization
- Distance calculations for QC

### Files Used
- `plot_velocity_model.py`: Visualize the velocity model
- `plot_simple_velocity_model.py`: Simplified visualization

### Process
1. Load model configuration
2. Generate cross-sectional and map views
3. Overlay source and receiver positions
4. Verify model dimensions and geometry

## 3. SPECFEM Meshing and Simulation

Run the seismic wave propagation simulation using SPECFEM3D.

### Key Components
- Mesh generation from model definition
- Database creation for material properties
- Wave propagation simulation
- Output collection and organization

### Files Used
- `src/simulation/specfem_runner.py`: Core simulation engine
- `run_specfem.py`: High-level simulation runner

### Process
1. Generate mesh with xmeshfem3D
2. Create databases with xgenerate_databases  
3. Run wave propagation with xspecfem3D
4. Collect seismograms (displacement/velocity)

## 4. DAS Data (Strain Rate) Conversion

Convert geophone data to DAS fiber strain rate measurements.

### Key Components
- Spatial derivative computation
- Gauge length simulation
- Multi-component integration
- Strain rate calculation

### Files Used
- `src/simulation/das_converter.py`: Core conversion library
- `convert_seismo_to_das.py`: Conversion script

### Process
1. Load velocity seismograms
2. Project onto fiber direction
3. Calculate spatial derivatives
4. Apply gauge length effect
5. Generate strain rate data

## 5. Geophone and DAS Data Visualization

Visualize and compare the geophone and DAS datasets.

### Key Components
- Shot gather visualization
- Wiggle plot display
- Waterfall plots for temporal evolution
- Component visualization (X, Y, Z, strain)

### Files Used
- `plot_shot_gather.py`: Create shot gather displays
- `plot_wavefield_snapshots.py`: Visualize wave propagation

### Process
1. Load processed data
2. Create multi-panel displays
3. Generate specialized visualizations (wiggle, waterfall)
4. Save publication-quality figures

## 6. Data Preparation and Model Training

Prepare datasets and train the ML models for geophone data interpolation.

### Key Components
- Dataset creation with various masking patterns
- Transformer model with multimodal encoding
- Training pipeline with validation
- Model evaluation

### Files Used
- `src/preprocessing/dataset.py`: Dataset preparation
- `src/models/transformer.py`: Model implementations
- `src/training/trainer.py`: Training workflows

### Process
1. Create masked datasets with different patterns
2. Configure transformer model architecture  
3. Train in stages (pretraining, finetuning)
4. Evaluate with multiple metrics
5. Visualize interpolation results

## Workflow Integration

The integrated workflow flows from model definition to final evaluation:

1. **Model Design**: Use the interactive Jupyter notebook to design and visualize the velocity model, source, and receivers
2. **Simulation**: Generate synthetic seismic data with SPECFEM3D
3. **Conversion**: Transform geophone data to DAS strain rate data
4. **Visualization**: Explore and compare the datasets
5. **Model Training**: Train transformer model to interpolate missing geophone data
6. **Evaluation**: Assess interpolation quality with metrics and visualizations

## Jupyter Notebook

The `notebooks/01_model_design_visualization.ipynb` notebook provides interactive visualization for:

- Model geometry and property definition
- Layer interface configuration
- Source parameter adjustment
- Receiver array design (geophone and DAS)
- Comprehensive model visualization
- Configuration file export

This notebook serves as the starting point for the entire workflow, allowing for visual inspection and parameter adjustment before running computationally expensive simulations.