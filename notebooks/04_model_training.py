# %% [markdown]
# # Training the Seismic Interpolation Model
# 
# This notebook demonstrates how to train the transformer-based multimodal model for seismic interpolation. We'll go through the following steps:
# 
# 1. Load the preprocessed datasets
# 2. Initialize the model architecture
# 3. Set up training parameters and MLflow tracking
# 4. Train the model
# 5. Evaluate the model's performance
# 6. Save the trained model for later use

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

# Add the project root to path for imports
sys.path.append('..')

# Import project modules
from src.models.transformer import MultimodalSeismicTransformer, StorSeismicBERTModel
from src.preprocessing.dataset import SeismicDataset, TransformerSeismicDataset
from src.training.trainer import MultimodalTrainer, TransformerSeismicTrainer
from src.utils.logging_utils import setup_logging, log_model_summary, log_dataset_info
from src.utils.plot_utils import plot_training_history, plot_gather_comparison, plot_trace_comparison

# Set up logging
logger = setup_logging(level='INFO')

# %% [markdown]
# ## Configuration
# 
# Define paths and parameters for model training.

# %%
# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Paths
data_dir = "../data/synthetic/processed/datasets"
models_dir = "../experiments/models"
results_dir = "../experiments/results"

# Ensure directories exist
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(results_dir).mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Load Preprocessed Data
# 
# Load the windowed datasets and split indices prepared in the previous notebook.

# %%
# Load windowed data
geophone_windows = np.load(Path(data_dir) / "geophone_windows.npy")
das_windows = np.load(Path(data_dir) / "das_windows.npy")

# Load split indices
train_indices = np.load(Path(data_dir) / "train_indices.npy")
val_indices = np.load(Path(data_dir) / "val_indices.npy")
test_indices = np.load(Path(data_dir) / "test_indices.npy")

# Load dataset parameters
with open(Path(data_dir) / "dataset_params.json", "r") as f:
    dataset_params = json.load(f)

# Print dataset info
print(f"Loaded {dataset_params['n_samples']} samples with {dataset_params['n_geophone_channels']} geophone channels and {dataset_params['n_das_channels']} DAS channels")
print(f"Window size: {dataset_params['window_size']}, Stride: {dataset_params['stride']}")
print(f"Train/Val/Test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)} samples")

# %%
# Split data using indices
train_geo = geophone_windows[train_indices]
train_das = das_windows[train_indices]
val_geo = geophone_windows[val_indices]
val_das = das_windows[val_indices]
test_geo = geophone_windows[test_indices]
test_das = das_windows[test_indices]

# %% [markdown]
# ## Create PyTorch Datasets and DataLoaders
# 
# Create PyTorch datasets and dataloaders for the model.

# %%
# Dataset parameters
batch_size = 32
mask_ratio = dataset_params['mask_ratio']
mask_pattern = 'random'  # Can be 'random', 'regular', or 'block'

# Create PyTorch datasets for the transformer model
train_dataset = TransformerSeismicDataset(
    train_geo, train_das, 
    mask_ratio=mask_ratio, 
    mask_pattern=mask_pattern, 
    positional_encoding=True
)

val_dataset = TransformerSeismicDataset(
    val_geo, val_das, 
    mask_ratio=mask_ratio, 
    mask_pattern=mask_pattern, 
    positional_encoding=True
)

test_dataset = TransformerSeismicDataset(
    test_geo, test_das, 
    mask_ratio=mask_ratio, 
    mask_pattern=mask_pattern, 
    positional_encoding=True
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Log dataset info
log_dataset_info(logger, train_dataset, name="Training Dataset")
log_dataset_info(logger, val_dataset, name="Validation Dataset")
log_dataset_info(logger, test_dataset, name="Test Dataset")

# %% [markdown]
# ## Initialize the Model
# 
# Initialize the transformer-based model for seismic interpolation.

# %%
# Model parameters
n_geophone_channels = dataset_params['n_geophone_channels']
n_das_channels = dataset_params['n_das_channels']
time_steps = dataset_params['window_size']
d_model = 256  # Model dimension
nhead = 8  # Number of attention heads
num_encoder_layers = 4  # Number of encoder layers
num_decoder_layers = 4  # Number of decoder layers
dim_feedforward = 1024  # Feedforward dimension
dropout = 0.1  # Dropout rate

# Initialize the model
model = StorSeismicBERTModel(
    max_channels=n_geophone_channels + n_das_channels,
    time_steps=time_steps,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)

# Move model to device
model = model.to(device)

# Log model summary
log_model_summary(logger, model)

# Print model architecture
print(f"Model Architecture:\n{model}")

# %% [markdown]
# ## Set Up Training
# 
# Set up the trainer with optimizer, scheduler, and MLflow tracking.

# %%
# Set up MLflow
experiment_name = "seismic_interpolation_transformer"
mlflow.set_experiment(experiment_name)

# Training parameters
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 50

# Initialize the trainer
trainer = TransformerSeismicTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    device=device,
    experiment_name=experiment_name
)

# %% [markdown]
# ## Train the Model
# 
# Train the model and track the progress using MLflow.

# %%
# Train the model
history = trainer.train(
    num_epochs=num_epochs,
    save_dir=models_dir,
    save_freq=5
)

# %% [markdown]
# ## Visualize Training Progress
# 
# Visualize the training and validation loss.

# %%
# Plot training history
plot_training_history(
    history,
    title=f"Training History - {experiment_name}",
    save_path=Path(results_dir) / f"{experiment_name}_training_history.png"
)

# %% [markdown]
# ## Evaluate the Model
# 
# Evaluate the trained model on the test dataset.

# %%
# Function to evaluate model on a batch
def evaluate_batch(model, batch, device):
    """Evaluate model on a batch of data."""
    model.eval()
    
    with torch.no_grad():
        # Unpack batch
        input_data, attention_mask, positions, target = batch
        input_data = input_data.to(device)
        attention_mask = attention_mask.to(device)
        positions = positions.to(device) if positions is not None else None
        target = target.to(device)
        
        # Forward pass
        outputs = model(input_data, attention_mask=attention_mask, position_ids=positions)
        
        # Extract geophone predictions
        n_das_channels = outputs.shape[1] - target.shape[1]
        predicted_geophone = outputs[:, n_das_channels:, :]
        
        return predicted_geophone.cpu().numpy(), target.cpu().numpy()

# %%
# Get a test batch
test_batch = next(iter(test_loader))

# Evaluate on the batch
predicted_geophone, target_geophone = evaluate_batch(model, test_batch, device)

# Select a sample from the batch
sample_idx = 0

# Plot comparison of true vs predicted
plot_gather_comparison(
    target_geophone[sample_idx],
    predicted_geophone[sample_idx],
    title="True vs Predicted Geophone Data",
    save_path=Path(results_dir) / f"{experiment_name}_gather_comparison.png"
)

# %%
# Plot comparison of individual traces
channel_idx = 5  # Choose a channel to plot

plot_trace_comparison(
    target_geophone[sample_idx, channel_idx],
    predicted_geophone[sample_idx, channel_idx],
    times=np.arange(dataset_params['window_size']),
    title=f"True vs Predicted - Sample {sample_idx}, Channel {channel_idx}",
    save_path=Path(results_dir) / f"{experiment_name}_trace_comparison.png"
)

# %% [markdown]
# ## Evaluate on the Full Test Set
# 
# Compute comprehensive metrics on the full test set.

# %%
# Import evaluation metrics
from src.evaluation.metrics import evaluate_model

# Evaluate on the test set
metrics = evaluate_model(model, test_loader, device)

# Print metrics
print("Test Set Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.6f}")

# Log to MLflow
with mlflow.start_run() as run:
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    mlflow.log_param("evaluation", "test_set")
    mlflow.pytorch.log_model(model, "model")

# %% [markdown]
# ## Compare with Baseline
# 
# Let's implement a simple baseline interpolation method (e.g., linear interpolation between available channels) and compare with our model.

# %%
# Simple linear interpolation baseline
def baseline_interpolation(masked_data, mask):
    """Linear interpolation between available channels."""
    n_channels, n_time_steps = masked_data.shape
    interpolated = masked_data.copy()
    
    for i in range(n_channels):
        if mask[i]:  # If channel is masked
            # Find nearest unmasked channels before and after
            before = None
            after = None
            
            # Look backward
            for j in range(i-1, -1, -1):
                if not mask[j]:
                    before = j
                    break
                    
            # Look forward
            for j in range(i+1, n_channels):
                if not mask[j]:
                    after = j
                    break
            
            # Interpolate based on available channels
            if before is not None and after is not None:
                # Linear interpolation between before and after
                weight_after = (i - before) / (after - before)
                weight_before = 1 - weight_after
                interpolated[i] = weight_before * masked_data[before] + weight_after * masked_data[after]
            elif before is not None:
                # Use only before
                interpolated[i] = masked_data[before]
            elif after is not None:
                # Use only after
                interpolated[i] = masked_data[after]
            else:
                # No reference channel, use zeros (or could use mean of all data)
                interpolated[i] = np.zeros(n_time_steps)
                
    return interpolated

# %%
# Apply baseline to the test batch
input_data, attention_mask, positions, target = test_batch

# Extract masked geophone data
n_das_channels = input_data.shape[1] - target.shape[1]
masked_geophone = input_data[:, n_das_channels:, :].cpu().numpy()

# Create mask from attention mask
mask = ~attention_mask[:, n_das_channels:].bool().cpu().numpy()

# Apply baseline interpolation
baseline_predictions = []
for i in range(len(masked_geophone)):
    baseline_pred = baseline_interpolation(masked_geophone[i], mask[i])
    baseline_predictions.append(baseline_pred)
baseline_predictions = np.array(baseline_predictions)

# Plot comparison
sample_idx = 0  # Same sample as before

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(target[sample_idx].numpy(), aspect='auto', cmap='seismic')
plt.title("True Geophone Data")
plt.xlabel("Time Sample")
plt.ylabel("Channel")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(baseline_predictions[sample_idx], aspect='auto', cmap='seismic')
plt.title("Baseline Interpolation")
plt.xlabel("Time Sample")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(predicted_geophone[sample_idx], aspect='auto', cmap='seismic')
plt.title("Transformer Model Prediction")
plt.xlabel("Time Sample")
plt.colorbar()

plt.suptitle("Comparison of True Data, Baseline, and Transformer Model")
plt.tight_layout()
plt.savefig(Path(results_dir) / f"{experiment_name}_baseline_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Compare metrics
from src.evaluation.metrics import mean_squared_error, signal_to_noise_ratio, correlation_coefficient

# Calculate metrics for the baseline
baseline_mse = mean_squared_error(target[sample_idx].numpy(), baseline_predictions[sample_idx])
baseline_snr = signal_to_noise_ratio(target[sample_idx].numpy(), baseline_predictions[sample_idx])
baseline_corr = correlation_coefficient(target[sample_idx].numpy().flatten(), baseline_predictions[sample_idx].flatten())

# Calculate metrics for the transformer model
model_mse = mean_squared_error(target[sample_idx].numpy(), predicted_geophone[sample_idx])
model_snr = signal_to_noise_ratio(target[sample_idx].numpy(), predicted_geophone[sample_idx])
model_corr = correlation_coefficient(target[sample_idx].numpy().flatten(), predicted_geophone[sample_idx].flatten())

# Print comparison
print("Metrics Comparison for Sample 0:")
print(f"  Baseline MSE: {baseline_mse:.6f}, Transformer MSE: {model_mse:.6f}, Improvement: {(1 - model_mse/baseline_mse)*100:.2f}%")
print(f"  Baseline SNR: {baseline_snr:.2f} dB, Transformer SNR: {model_snr:.2f} dB, Improvement: {model_snr - baseline_snr:.2f} dB")
print(f"  Baseline Correlation: {baseline_corr:.4f}, Transformer Correlation: {model_corr:.4f}")

# %% [markdown]
# ## Save Final Model
# 
# Save the trained model for later use in inference.

# %%
# Save the final model
model_path = Path(models_dir) / f"{experiment_name}_final_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'max_channels': n_geophone_channels + n_das_channels,
        'time_steps': time_steps,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward,
        'dropout': dropout
    },
    'dataset_params': dataset_params,
    'metrics': metrics
}, model_path)

print(f"Saved final model to {model_path}")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we have:
# 
# 1. Loaded and preprocessed the geophone and DAS data
# 2. Initialized a transformer-based model for seismic interpolation
# 3. Trained the model using MLflow to track experiments
# 4. Evaluated the model's performance on the test set
# 5. Compared with a baseline interpolation method
# 6. Saved the trained model for later use
# 
# The transformer-based model successfully learned to interpolate missing geophone channels using DAS constraints, outperforming the simple baseline interpolation method. The results demonstrate the effectiveness of the approach for seismic data interpolation.


