# SPECFEM3D Simulation and Visualization Toolkit

A comprehensive toolkit for running SPECFEM3D simulations and generating high-quality visualizations of seismic wave propagation. This toolkit provides a streamlined workflow from parameter management to simulation execution and data visualization.

## Features

### Simulation Management
- Run SPECFEM3D simulations with predefined parameter sets
- Flexible processor configuration
- Automatic parameter validation and management
- Support for different simulation configurations (standard, high-resolution, DAS fiber)

### Visualization Capabilities
1. **Shot Gather Plots**
   - Display seismic wave recordings at each receiver over time
   - Support for velocity, displacement, and acceleration data
   - Both color and wiggle trace representations
   - Automatic normalization and scaling options

2. **Wavefield Snapshots**
   - Generate static snapshots of wave propagation at different time points
   - Show all three components (X, Y, Z) side by side
   - Grid layout for easy comparison of wave states
   - High-resolution output suitable for publication

3. **Combined Workflow**
   - Run simulation and generate visualizations in one command
   - Automatic output organization
   - Progress monitoring and timeout handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml_interpolation.git
cd ml_interpolation
```

2. Ensure you have the required Python packages:
```bash
pip install numpy matplotlib
```

3. Make sure SPECFEM3D is installed and properly configured on your system.

## Usage

### Basic Usage

Run a simulation with default parameters and generate visualizations:
```bash
python simulate_and_visualize.py --specfem-dir ~/specfem3d --output-dir ./plots
```

### Using Parameter Sets

Run a simulation with a predefined parameter set:
```bash
python simulate_and_visualize.py --parameter-set parameter_sets/standard_simulation.json
```

### Customizing Visualizations

Generate visualizations for specific data types:
```bash
python simulate_and_visualize.py --data-types velocity displacement acceleration
```

### Visualization Only

Generate visualizations from existing simulation results:
```bash
python simulate_and_visualize.py --skip-simulation --output-dir ./plots
```

## Scripts Overview

### `simulate_and_visualize.py`
Main script that orchestrates the simulation and visualization process.

### `plot_shot_gather.py`
Generates shot gather plots showing seismic wave recordings at each receiver.

### `plot_wavefield_snapshots.py`
Creates static snapshots of wave propagation at different time points.

### `param_manager.py`
Manages simulation parameters and configuration files.

## Output Files

The toolkit generates the following types of output files:

1. **Shot Gather Plots**
   - `shot_gather_X_velocity.png`
   - `shot_gather_wiggle_X_velocity.png`
   - (Similar files for Y and Z components)

2. **Wavefield Snapshots**
   - `wavefield_snapshots_velocity.png`
   - `wavefield_snapshots_displacement.png`

## Parameter Sets

Predefined parameter sets are available in the `parameter_sets` directory:
- `standard_simulation.json`: Balanced resolution/performance
- `high_res_simulation.json`: Higher resolution for detailed analysis
- `fast_simulation.json`: Quick tests with lower resolution
- `das_fiber_simulation.json`: Optimized for DAS fiber simulations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SPECFEM3D development team for the core simulation software
- Contributors and users who have provided feedback and suggestions