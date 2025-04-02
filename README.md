# ML Interpolation Project

A comprehensive toolkit for seismic data simulation, DAS fiber conversion, and machine learning-based interpolation. This project provides a streamlined workflow from parameter management to simulation execution, data processing, model training, and evaluation.

## Features

### Simulation Management
- Run SPECFEM3D simulations with unified parameter management
- Convert geophone seismic data to Distributed Acoustic Sensing (DAS) measurements
- Flexible processor configuration with automatic parameter validation

### Machine Learning Components
- Transformer-based models for interpolating missing geophone data using DAS constraints
- Customizable masking patterns to simulate various types of missing data scenarios
- Advanced training workflows with validation and evaluation metrics

### Visualization Capabilities
- Interactive model design and visualization through Jupyter notebooks
- Shot gather plots with support for different data types and representations
- Wavefield snapshots showing the evolution of wave propagation
- Comparison visualizations between original and interpolated data

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

3. Ensure SPECFEM3D is installed and properly configured on your system.

## Project Structure

```
ml_interpolation/
├── workflow.py                # Main workflow orchestration script
├── param_manager.py           # Parameter management utilities
├── parameter_sets/            # Predefined parameter sets
│   ├── default.json           # Default configuration
│   └── ...
├── src/                       # Source code modules
│   ├── simulation/            # Simulation-related code
│   │   ├── specfem_runner.py  # SPECFEM execution engine
│   │   └── das_converter.py   # DAS conversion utilities
│   ├── preprocessing/         # Data preprocessing utilities
│   ├── models/                # ML model architectures
│   ├── training/              # Training workflows
│   └── evaluation/            # Evaluation metrics and tools
├── notebooks/                 # Jupyter notebooks
│   ├── 01_model_design_visualization.ipynb  # Interactive model design
│   └── 05_evaluation_and_visualization.ipynb  # Results visualization
├── plot_velocity_model.py     # Velocity model visualization
├── plot_shot_gather.py        # Shot gather visualization
├── plot_wavefield_snapshots.py  # Wavefield snapshot visualization
└── convert_seismo_to_das.py   # Standalone DAS conversion tool
```

## Workflow

The streamlined workflow follows these steps:

1. **Parameter Configuration**: Define simulation parameters using JSON configuration files
2. **Model Design**: Use Jupyter notebooks for interactive model design and visualization
3. **SPECFEM Simulation**: Run seismic wave propagation simulation with SPECFEM3D
4. **DAS Conversion**: Convert geophone data to DAS fiber strain rate measurements
5. **Data Visualization**: Generate visualizations of simulation results
6. **Data Preprocessing**: Prepare datasets for model training
7. **Model Training**: Train transformer model for geophone data interpolation
8. **Evaluation**: Assess model performance with various metrics

## Usage

### Complete Workflow

Run the entire workflow from simulation to evaluation:

```bash
python workflow.py --config parameter_sets/default.json --specfem-dir /path/to/specfem3d --run-all
```

### Individual Steps

Run specific components of the workflow:

```bash
# Run only the simulation
python workflow.py --config parameter_sets/default.json --specfem-dir /path/to/specfem3d --run-simulation

# Convert existing simulation data to DAS
python workflow.py --config parameter_sets/default.json --convert-das

# Visualize data
python workflow.py --config parameter_sets/default.json --visualize-data

# Train the model
python workflow.py --config parameter_sets/default.json --train-model

# Evaluate model performance
python workflow.py --config parameter_sets/default.json --evaluate-model
```

### Interactive Model Design

Use the Jupyter notebook for interactive model design:

```bash
jupyter notebook notebooks/01_model_design_visualization.ipynb
```

## Parameter Sets

The `parameter_sets` directory contains predefined parameter configurations:

- `default.json`: Standard configuration with balanced settings
- `high_resolution.json`: Higher resolution for detailed analysis
- `fast_simulation.json`: Quicker simulation with lower resolution for testing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SPECFEM3D development team for the core simulation software
- PyTorch team for the deep learning framework
- Contributors and users who have provided feedback and suggestions