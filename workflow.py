#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Interpolation Workflow
-------------------------
Main script to orchestrate the entire workflow for seismic data simulation,
DAS conversion, and ML-based interpolation.
"""

import os
import argparse
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Import modules from src
from src.simulation.specfem_runner import SpecfemRunner
from src.simulation.das_converter import DASConverter
from src.preprocessing.dataset import SeismicDataset
from src.models.transformer import MultimodalSeismicTransformer
from src.training.trainer import SeismicModelTrainer
from src.evaluation.metrics import evaluate_interpolation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_arg_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(description='ML Interpolation Workflow')
    
    # General options
    parser.add_argument('--config', type=str, default='parameter_sets/default.json',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to store all outputs')
    parser.add_argument('--specfem-dir', type=str, 
                        help='Path to SPECFEM3D installation')
    
    # Stage selection
    parser.add_argument('--run-simulation', action='store_true',
                        help='Run SPECFEM simulation')
    parser.add_argument('--convert-das', action='store_true',
                        help='Convert geophone data to DAS')
    parser.add_argument('--visualize-data', action='store_true',
                        help='Create visualizations of data')
    parser.add_argument('--train-model', action='store_true',
                        help='Train ML interpolation model')
    parser.add_argument('--evaluate-model', action='store_true',
                        help='Evaluate model performance')
    parser.add_argument('--run-all', action='store_true',
                        help='Run the complete workflow')
    
    return parser

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_simulation(config, specfem_dir, output_dir):
    """Run SPECFEM simulation with the given configuration."""
    logger.info("Running SPECFEM simulation...")
    
    runner = SpecfemRunner(
        specfem_dir=specfem_dir,
        output_dir=output_dir,
        config=config
    )
    
    # Execute simulation steps
    runner.prepare_simulation()
    runner.run_mesher()
    runner.run_database_generator()
    runner.run_solver()
    
    # Collect and organize outputs
    seismogram_files = runner.collect_output_files()
    
    logger.info(f"Simulation completed. Output files: {len(seismogram_files)}")
    return seismogram_files

def convert_to_das(config, seismogram_files, output_dir):
    """Convert geophone data to DAS strain rate measurements."""
    logger.info("Converting geophone data to DAS measurements...")
    
    converter = DASConverter(
        config=config,
        output_dir=output_dir
    )
    
    das_files = converter.convert_files(seismogram_files)
    
    logger.info(f"DAS conversion completed. Output files: {len(das_files)}")
    return das_files

def visualize_data(config, geophone_files, das_files, output_dir):
    """Create visualizations of geophone and DAS data."""
    logger.info("Creating data visualizations...")
    
    # Import visualization modules only when needed
    from plot_shot_gather import plot_shot_gather
    from plot_wavefield_snapshots import plot_wavefield_snapshots
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Shot gather plots
    shot_gather_file = os.path.join(plots_dir, 'shot_gather_comparison.png')
    plot_shot_gather(
        geophone_files=geophone_files,
        das_files=das_files,
        output_file=shot_gather_file,
        config=config
    )
    
    # Wavefield snapshots
    wavefield_file = os.path.join(plots_dir, 'wavefield_snapshots.png')
    plot_wavefield_snapshots(
        data_files=geophone_files,
        output_file=wavefield_file,
        config=config
    )
    
    logger.info(f"Visualizations created in {plots_dir}")
    return plots_dir

def prepare_datasets(config, geophone_files, das_files, output_dir):
    """Prepare datasets for model training and evaluation."""
    logger.info("Preparing datasets for ML model...")
    
    # Create dataset instances
    dataset = SeismicDataset(
        geophone_files=geophone_files,
        das_files=das_files,
        config=config,
        output_dir=output_dir
    )
    
    # Generate train/val/test splits with masking
    train_dataset, val_dataset, test_dataset = dataset.create_datasets()
    
    logger.info(f"Datasets prepared: Train:{len(train_dataset)}, Val:{len(val_dataset)}, Test:{len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

def train_model(config, train_dataset, val_dataset, output_dir):
    """Train the interpolation model."""
    logger.info("Training ML interpolation model...")
    
    # Initialize model
    model = MultimodalSeismicTransformer(config=config)
    
    # Initialize trainer
    trainer = SeismicModelTrainer(
        model=model,
        config=config,
        output_dir=output_dir
    )
    
    # Train model
    trained_model = trainer.train(train_dataset, val_dataset)
    
    model_path = os.path.join(output_dir, 'models', 'trained_model.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)
    
    logger.info(f"Model training completed. Model saved to {model_path}")
    return trained_model, model_path

def evaluate_model(config, model, test_dataset, output_dir):
    """Evaluate model performance."""
    logger.info("Evaluating model performance...")
    
    # Run evaluation
    metrics, predictions = evaluate_interpolation(
        model=model,
        test_dataset=test_dataset,
        config=config
    )
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'evaluation', 'metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualization of results
    results_path = os.path.join(output_dir, 'evaluation', 'interpolation_results.png')
    # TODO: Add visualization code
    
    logger.info(f"Evaluation completed. Metrics saved to {metrics_path}")
    return metrics, predictions

def main():
    """Main workflow execution function."""
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # If --run-all is specified, enable all stages
    if args.run_all:
        args.run_simulation = True
        args.convert_das = True
        args.visualize_data = True
        args.train_model = True
        args.evaluate_model = True
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables to store outputs from each stage
    seismogram_files = None
    das_files = None
    train_dataset, val_dataset, test_dataset = None, None, None
    trained_model = None
    
    # Run selected workflow stages
    if args.run_simulation:
        seismogram_files = run_simulation(config, args.specfem_dir, output_dir)
    
    if args.convert_das:
        if seismogram_files is None:
            # Try to load from previous run
            sim_output_dir = os.path.join(output_dir, 'simulation')
            seismogram_files = [f for f in os.listdir(sim_output_dir) if f.endswith('.semd')]
        
        das_files = convert_to_das(config, seismogram_files, output_dir)
    
    if args.visualize_data:
        if seismogram_files is None or das_files is None:
            # Try to load from previous run
            sim_output_dir = os.path.join(output_dir, 'simulation')
            das_output_dir = os.path.join(output_dir, 'das')
            
            if seismogram_files is None:
                seismogram_files = [f for f in os.listdir(sim_output_dir) if f.endswith('.semd')]
            
            if das_files is None:
                das_files = [f for f in os.listdir(das_output_dir) if f.endswith('.das')]
        
        visualize_data(config, seismogram_files, das_files, output_dir)
    
    if args.train_model or args.evaluate_model:
        if seismogram_files is None or das_files is None:
            # Try to load from previous run
            sim_output_dir = os.path.join(output_dir, 'simulation')
            das_output_dir = os.path.join(output_dir, 'das')
            
            if seismogram_files is None:
                seismogram_files = [f for f in os.listdir(sim_output_dir) if f.endswith('.semd')]
            
            if das_files is None:
                das_files = [f for f in os.listdir(das_output_dir) if f.endswith('.das')]
        
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            config, seismogram_files, das_files, output_dir
        )
    
    if args.train_model:
        trained_model, model_path = train_model(config, train_dataset, val_dataset, output_dir)
    
    if args.evaluate_model:
        if trained_model is None:
            # Try to load from previous run
            model_path = os.path.join(output_dir, 'models', 'trained_model.pt')
            if os.path.exists(model_path):
                # Initialize model
                model = MultimodalSeismicTransformer(config=config)
                # Load weights
                model.load_state_dict(torch.load(model_path))
                trained_model = model
        
        metrics, predictions = evaluate_model(config, trained_model, test_dataset, output_dir)
    
    logger.info("Workflow completed successfully!")

if __name__ == "__main__":
    main() 