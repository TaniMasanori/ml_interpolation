# %% [markdown]
# # Data Preprocessing for Seismic Interpolation
# 
# This notebook demonstrates how to preprocess the synthetic geophone and DAS data for training the seismic interpolation model. We'll go through the following steps:
# 
# 1. Load the combined geophone and DAS dataset
# 2. Preprocess and normalize the data
# 3. Create masked datasets for training (simulating missing geophone channels)
# 4. Prepare windowed data samples for the model
# 5. Split the data into training, validation, and test sets

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add the project root to path for imports
sys.path.append('..')

# Import project modules
from src.preprocessing.data_loader import normalize_data, create_masked_dataset, prepare_dataset_for_training, split_dataset
from src.preprocessing.dataset import SeismicDataset, TransformerSeismicDataset
from src.utils.logging_utils import setup_logging
from src.utils.plot_utils import plot_seismic_gather, plot_masked_gather

# Set up logging
logger = setup_logging(level='INFO')

# %% [markdown]
# ## Load Combined Dataset
# 
# First, let's load the combined dataset of geophone and DAS data that we created in the previous notebook.

# %%
# Load combined dataset
dataset_path = "../data/synthetic/processed/simulation1/combined_dataset.npy"
combined_data = np.load(dataset_path, allow_pickle=True).item()

# Extract data
times = combined_data["times"]
geophone_data = combined_data["geophone_data"]
das_data = combined_data["das_data"]
station_coords = combined_data["station_coordinates"]
metadata = combined_data["metadata"]

# Print dataset info
print(f"Loaded dataset with {metadata['n_geophones']} geophones and {metadata['n_das_channels']} DAS channels")
print(f"Each trace has {len(times)} time samples, dt = {metadata['dt']} s")
print(f"DAS gauge length: {metadata['gauge_length']} m, channel spacing: {metadata['channel_spacing']} m")

# %% [markdown]
# ## Visualize Raw Data
# 
# Let's visualize the raw geophone and DAS data to confirm they're loaded correctly.

# %%
# Plot the geophone data
plot_seismic_gather(geophone_data, title="Geophone Data", xlabel="Time Sample", ylabel="Channel")

# %%
# Plot the DAS data
plot_seismic_gather(das_data, title="DAS Data", xlabel="Time Sample", ylabel="Channel")

# %% [markdown]
# ## Preprocess and Normalize Data
# 
# Now, let's preprocess and normalize the data to prepare it for the model.

# %%
# Normalize data (channel-wise)
norm_geophone_data = normalize_data(geophone_data, method='max')
norm_das_data = normalize_data(das_data, method='max')

# Plot normalized data
plot_seismic_gather(norm_geophone_data, title="Normalized Geophone Data", xlabel="Time Sample", ylabel="Channel")

# %%
plot_seismic_gather(norm_das_data, title="Normalized DAS Data", xlabel="Time Sample", ylabel="Channel")

# %% [markdown]
# ## Create Masked Dataset
# 
# Now let's create a masked dataset to simulate missing geophone channels. This is the core of our interpolation task - we'll train the model to reconstruct these missing channels using the available geophone data and DAS constraints.

# %%
# Create a masked dataset with different patterns
# Try different masking patterns
mask_patterns = ['random', 'regular', 'block']
mask_ratio = 0.3  # 30% of channels masked

for pattern in mask_patterns:
    # Create masked dataset
    masked_geo_data, mask, target_geo_data = create_masked_dataset(
        norm_geophone_data, norm_das_data, 
        mask_pattern=pattern, 
        mask_ratio=mask_ratio
    )
    
    # Plot the masked data
    plot_masked_gather(
        norm_geophone_data, 
        mask, 
        title=f"Masked Geophone Data ({pattern} pattern, {int(mask_ratio*100)}% masked)", 
        cmap='seismic'
    )

# %% [markdown]
# ## Prepare Windowed Data for Training
# 
# For training, we'll create windowed samples of fixed size from the data.

# %%
# Create windowed datasets
window_size = 256  # Number of time steps per window
stride = 128  # Stride between windows

# Prepare windowed data
geophone_windows, das_windows = prepare_dataset_for_training(
    times, norm_geophone_data, norm_das_data, 
    window_size=window_size, 
    stride=stride
)

print(f"Created {geophone_windows.shape[0]} windows of size {window_size}")
print(f"Geophone windows shape: {geophone_windows.shape}")
print(f"DAS windows shape: {das_windows.shape}")

# %%
# Plot an example windowed data sample
sample_idx = 5  # Example window index

# Plot geophone window
plt.figure(figsize=(10, 6))
plt.imshow(geophone_windows[sample_idx], aspect='auto', cmap='seismic')
plt.title(f"Geophone Window {sample_idx}")
plt.xlabel("Time Sample")
plt.ylabel("Channel")
plt.colorbar()
plt.tight_layout()
plt.show()

# Plot DAS window
plt.figure(figsize=(10, 6))
plt.imshow(das_windows[sample_idx], aspect='auto', cmap='seismic')
plt.title(f"DAS Window {sample_idx}")
plt.xlabel("Time Sample")
plt.ylabel("Channel")
plt.colorbar()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Split Data into Training, Validation, and Test Sets
# 
# Now let's split the windowed data into training, validation, and test sets.

# %%
# Split data
split_ratio = (0.7, 0.15, 0.15)  # 70% train, 15% validation, 15% test
train_data, val_data, test_data = split_dataset(geophone_windows, das_windows, split_ratio=split_ratio)

# Unpack the data
train_geo, train_das = train_data
val_geo, val_das = val_data
test_geo, test_das = test_data

print(f"Training set: {train_geo.shape[0]} samples")
print(f"Validation set: {val_geo.shape[0]} samples")
print(f"Test set: {test_geo.shape[0]} samples")

# %% [markdown]
# ## Create PyTorch Datasets and DataLoaders
# 
# Let's create PyTorch datasets and dataloaders for both the standard model and the transformer model.

# %%
# Create PyTorch datasets
batch_size = 32
mask_ratio = 0.3
mask_pattern = 'random'

# Standard dataset for multimodal model
train_dataset = SeismicDataset(train_geo, train_das, mask_ratio=mask_ratio, mask_pattern=mask_pattern)
val_dataset = SeismicDataset(val_geo, val_das, mask_ratio=mask_ratio, mask_pattern=mask_pattern)
test_dataset = SeismicDataset(test_geo, test_das, mask_ratio=mask_ratio, mask_pattern=mask_pattern)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Created DataLoaders with batch size {batch_size}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# %%
# Create transformer dataset
train_transformer_dataset = TransformerSeismicDataset(
    train_geo, train_das, 
    mask_ratio=mask_ratio, 
    mask_pattern=mask_pattern, 
    positional_encoding=True
)

val_transformer_dataset = TransformerSeismicDataset(
    val_geo, val_das, 
    mask_ratio=mask_ratio, 
    mask_pattern=mask_pattern, 
    positional_encoding=True
)

test_transformer_dataset = TransformerSeismicDataset(
    test_geo, test_das, 
    mask_ratio=mask_ratio, 
    mask_pattern=mask_pattern, 
    positional_encoding=True
)

# Create transformer dataloaders
train_transformer_loader = DataLoader(train_transformer_dataset, batch_size=batch_size, shuffle=True)
val_transformer_loader = DataLoader(val_transformer_dataset, batch_size=batch_size, shuffle=False)
test_transformer_loader = DataLoader(test_transformer_dataset, batch_size=batch_size, shuffle=False)

print(f"Created Transformer DataLoaders with batch size {batch_size}")
print(f"Training batches: {len(train_transformer_loader)}")
print(f"Validation batches: {len(val_transformer_loader)}")
print(f"Test batches: {len(test_transformer_loader)}")

# %% [markdown]
# ## Examine a Batch from the Dataset
# 
# Let's examine a batch from the dataset to verify it's structured correctly.

# %%
# Get a batch from the standard dataset
for batch in train_loader:
    masked_geophone, das, mask, target_geophone = batch
    
    print(f"Masked Geophone shape: {masked_geophone.shape}")
    print(f"DAS shape: {das.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Target Geophone shape: {target_geophone.shape}")
    
    # Count masked channels in first sample
    masked_count = mask[0].sum().item()
    print(f"Sample 0 has {masked_count} masked channels out of {mask[0].shape[0]} ({masked_count/mask[0].shape[0]*100:.1f}%)")
    
    # Visualize first sample in batch
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.imshow(masked_geophone[0].cpu().numpy(), aspect='auto', cmap='seismic')
    plt.title("Masked Geophone (Input)")
    plt.ylabel("Channel")
    plt.colorbar()
    
    plt.subplot(3, 1, 2)
    plt.imshow(das[0].cpu().numpy(), aspect='auto', cmap='seismic')
    plt.title("DAS Data (Input)")
    plt.ylabel("Channel")
    plt.colorbar()
    
    plt.subplot(3, 1, 3)
    plt.imshow(target_geophone[0].cpu().numpy(), aspect='auto', cmap='seismic')
    plt.title("Target Geophone (Complete)")
    plt.xlabel("Time Sample")
    plt.ylabel("Channel")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    break

# %%
# Get a batch from the transformer dataset
for batch in train_transformer_loader:
    input_data, attention_mask, positions, target = batch
    
    print(f"Input Data shape: {input_data.shape}")
    print(f"Attention Mask shape: {attention_mask.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Target shape: {target.shape}")
    
    # Visualize first sample in batch
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.imshow(input_data[0].cpu().numpy(), aspect='auto', cmap='seismic')
    plt.title("Input Data (Concatenated DAS + Masked Geophone)")
    plt.ylabel("Channel")
    plt.colorbar()
    
    plt.subplot(2, 1, 2)
    plt.imshow(target[0].cpu().numpy(), aspect='auto', cmap='seismic')
    plt.title("Target (Complete Geophone)")
    plt.xlabel("Time Sample")
    plt.ylabel("Channel")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Visualize attention mask
    plt.figure(figsize=(10, 3))
    plt.imshow(attention_mask[0].cpu().numpy().reshape(1, -1), aspect='auto', cmap='binary')
    plt.title("Attention Mask (1 = attend, 0 = ignore)")
    plt.xlabel("Channel")
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    break

# %% [markdown]
# ## Save Processed Datasets for Training
# 
# Finally, let's save the processed datasets for subsequent model training.

# %%
# Save the processed data
processed_dir = Path("../data/synthetic/processed/datasets")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save windowed data
np.save(processed_dir / "geophone_windows.npy", geophone_windows)
np.save(processed_dir / "das_windows.npy", das_windows)

# Save train/val/test split indices
n_samples = geophone_windows.shape[0]
indices = np.random.permutation(n_samples)
train_end = int(split_ratio[0] * n_samples)
val_end = train_end + int(split_ratio[1] * n_samples)
train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

np.save(processed_dir / "train_indices.npy", train_indices)
np.save(processed_dir / "val_indices.npy", val_indices)
np.save(processed_dir / "test_indices.npy", test_indices)

# Save dataset parameters
dataset_params = {
    "window_size": window_size,
    "stride": stride,
    "n_samples": n_samples,
    "n_geophone_channels": geophone_windows.shape[1],
    "n_das_channels": das_windows.shape[1],
    "mask_ratio": mask_ratio,
    "split_ratio": split_ratio,
    "metadata": metadata
}

# Save as JSON
import json
with open(processed_dir / "dataset_params.json", "w") as f:
    json.dump(dataset_params, f, indent=2)

print(f"Saved processed datasets to {processed_dir}")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we have:
# 
# 1. Loaded the combined geophone and DAS dataset
# 2. Preprocessed and normalized the data
# 3. Created masked datasets for training (simulating missing geophone channels)
# 4. Prepared windowed data samples for the model
# 5. Split the data into training, validation, and test sets
# 6. Created PyTorch datasets and dataloaders for both standard and transformer models
# 7. Saved the processed datasets for subsequent model training
# 
# The data is now ready for training the seismic interpolation model, which we'll do in the next notebook.


