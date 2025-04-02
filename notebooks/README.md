# ML Interpolation Workflow Notebooks

This directory contains Jupyter notebooks that provide interactive components for the ML Interpolation project workflow.

## Notebooks Overview

### 1. [01_model_design_visualization.ipynb](./01_model_design_visualization.ipynb)
- Interactive design of velocity models, sources, and receivers
- Visualization of model properties and geometry
- Parameter adjustment and validation
- Generation of SPECFEM3D configuration files

### 2. [05_evaluation_and_visualization.ipynb](./05_evaluation_and_visualization.ipynb)
- Visualization of interpolation results
- Comparison between original, masked, and predicted data
- Evaluation metrics visualization
- Frequency domain analysis

## Using the Workflow

The complete workflow follows these steps:

1. **Model Design**: Use `01_model_design_visualization.ipynb` to design your model interactively
2. **Simulation**: Run `workflow.py --run-simulation` to execute SPECFEM3D simulation
3. **DAS Conversion**: Run `workflow.py --convert-das` to convert geophone data to DAS
4. **Visualization**: Run `workflow.py --visualize-data` to create data visualizations
5. **Model Training**: Run `workflow.py --train-model` to train the interpolation model
6. **Evaluation**: Use `05_evaluation_and_visualization.ipynb` to visualize and analyze results

## Command-Line Workflow

The workflow can also be executed from the command line using:

```bash
# Run the complete workflow
python workflow.py --config parameter_sets/default.json --specfem-dir ~/specfem3d --run-all

# Run individual steps
python workflow.py --config parameter_sets/default.json --specfem-dir ~/specfem3d --run-simulation
python workflow.py --config parameter_sets/default.json --convert-das
python workflow.py --config parameter_sets/default.json --visualize-data
python workflow.py --config parameter_sets/default.json --train-model
python workflow.py --config parameter_sets/default.json --evaluate-model
```

## Streamlined Framework

The notebooks are integrated with the streamlined workflow, which uses:
- `param_manager.py` as the single source for parameter management
- `src/simulation/specfem_runner.py` for all simulation steps
- `src/simulation/das_converter.py` for DAS conversion
- The training components in `src/models`, `src/training`, and `src/evaluation`

This ensures a consistent parameter space and eliminates redundant code across the workflow.