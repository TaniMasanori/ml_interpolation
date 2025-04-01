"""
Logging utilities for the seismic interpolation project.

This module provides functions to set up logging for the project.
"""
import logging
import sys
from pathlib import Path
import datetime

def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file (str, optional): Path to log file. If None, logs only to console.
        level (int, optional): Logging level, e.g., logging.INFO, logging.DEBUG.
        
    Returns:
        logging.Logger: Root logger
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_experiment_logger(experiment_name):
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        logging.Logger: Logger for the experiment
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # If logger already has handlers, don't add more
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Started logging for experiment: {experiment_name}")
    
    return logger

def log_config(logger, config):
    """
    Log configuration parameters.
    
    Args:
        logger (logging.Logger): Logger instance
        config (dict): Configuration parameters
    """
    logger.info("Configuration parameters:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

def log_metrics(logger, metrics, prefix=''):
    """
    Log evaluation metrics.
    
    Args:
        logger (logging.Logger): Logger instance
        metrics (dict): Evaluation metrics
        prefix (str, optional): Prefix to add to metric names
    """
    logger.info(f"{prefix}Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

def log_model_summary(logger, model):
    """
    Log model summary.
    
    Args:
        logger (logging.Logger): Logger instance
        model (torch.nn.Module): PyTorch model
    """
    logger.info("Model summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Log model architecture (simplified)
    logger.info("  Model architecture:")
    for name, module in model.named_children():
        logger.info(f"    {name}: {module.__class__.__name__}")
        
        # Log submodules if available
        for subname, submodule in module.named_children():
            logger.info(f"      {subname}: {submodule.__class__.__name__}")

def log_dataset_info(logger, dataset, name='Dataset'):
    """
    Log dataset information.
    
    Args:
        logger (logging.Logger): Logger instance
        dataset: Dataset instance
        name (str, optional): Dataset name
    """
    logger.info(f"{name} information:")
    logger.info(f"  Number of samples: {len(dataset)}")
    
    # Log specific dataset attributes if available
    if hasattr(dataset, 'n_channels'):
        logger.info(f"  Number of geophone channels: {dataset.n_channels}")
    if hasattr(dataset, 'n_das_channels'):
        logger.info(f"  Number of DAS channels: {dataset.n_das_channels}")
    if hasattr(dataset, 'n_time_steps'):
        logger.info(f"  Number of time steps: {dataset.n_time_steps}")
    if hasattr(dataset, 'mask_ratio'):
        logger.info(f"  Mask ratio: {dataset.mask_ratio}")
    if hasattr(dataset, 'mask_pattern'):
        logger.info(f"  Mask pattern: {dataset.mask_pattern}")

def log_simulation_params(logger, params):
    """
    Log simulation parameters.
    
    Args:
        logger (logging.Logger): Logger instance
        params (dict): Simulation parameters
    """
    logger.info("Simulation parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

def log_das_conversion_params(logger, params):
    """
    Log DAS conversion parameters.
    
    Args:
        logger (logging.Logger): Logger instance
        params (dict): DAS conversion parameters
    """
    logger.info("DAS conversion parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

def log_mlflow_run_info(logger, run_id, experiment_id, artifact_uri):
    """
    Log MLflow run information.
    
    Args:
        logger (logging.Logger): Logger instance
        run_id (str): MLflow run ID
        experiment_id (str): MLflow experiment ID
        artifact_uri (str): MLflow artifact URI
    """
    logger.info("MLflow run information:")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Experiment ID: {experiment_id}")
    logger.info(f"  Artifact URI: {artifact_uri}")
    logger.info(f"  Tracking URL: http://localhost:5000 (if MLflow server is running locally)")

def log_training_progress(logger, epoch, metrics, lr=None):
    """
    Log training progress.
    
    Args:
        logger (logging.Logger): Logger instance
        epoch (int): Current epoch
        metrics (dict): Training metrics
        lr (float, optional): Current learning rate
    """
    msg = f"Epoch {epoch}"
    
    for key, value in metrics.items():
        msg += f", {key}: {value:.6f}"
        
    if lr is not None:
        msg += f", lr: {lr:.8f}"
        
    logger.info(msg)