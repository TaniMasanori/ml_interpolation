# %% [markdown]
# # DAS Conversion: From Particle Velocity to Strain Rate
# 
# This notebook demonstrates how to convert SPECFEM3D particle velocity outputs to Distributed Acoustic Sensing (DAS) strain-rate responses. We'll go through the following steps:
# 
# 1. Load synthetic seismic data generated by SPECFEM3D
# 2. Apply the DAS conversion algorithm to simulate DAS strain-rate measurements
# 3. Visualize and compare geophone and DAS data
# 4. Save the processed DAS data for use in the interpolation model

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add the project root to path for imports
sys.path.append('..')

# Import project modules
from src.simulation.das_converter import DASConverter
from src.utils.logging_utils import setup_logging
from src.utils.plot_utils import plot_seismic_trace, plot_seismic_gather, plot_multiple_traces

# Set up logging
logger = setup_logging(level='INFO')

# %% [markdown]
# ## Configuration
# 
# Define paths and parameters for the DAS conversion.

# %%
# Paths
raw_data_dir = "../data/synthetic/raw/simulation1"  # Raw simulation outputs
processed_geo_dir = "../data/synthetic/processed/simulation1"  # Processed geophone data
processed_das_dir = "../data/synthetic/processed/simulation1/das"  # Directory for DAS data

# Ensure output directory exists
Path(processed_das_dir).mkdir(parents=True, exist_ok=True)

# DAS parameters
gauge_length = 10.0  # Gauge length in meters
channel_spacing = 10.0  # Spacing between DAS channels in meters

# %% [markdown]
# ## Load Synthetic Geophone Data
# 
# Load the synthetic seismic data generated by SPECFEM3D. We'll use the particle velocity data as the basis for DAS conversion.

# %%
# Check if processed data exists
if Path(processed_geo_dir).exists():
    # Load processed data
    times = np.load(Path(processed_geo_dir) / "times.npy")
    data_x = np.load(Path(processed_geo_dir) / "data_x.npy")
    data_y = np.load(Path(processed_geo_dir) / "data_y.npy")
    data_z = np.load(Path(processed_geo_dir) / "data_z.npy")
    station_df = pd.read_csv(Path(processed_geo_dir) / "stations.csv")
    
    print(f"Loaded {data_x.shape[0]} X-component seismograms, each with {data_x.shape[1]} time steps.")
    print(f"Loaded {data_y.shape[0]} Y-component seismograms, each with {data_y.shape[1]} time steps.")
    print(f"Loaded {data_z.shape[0]} Z-component seismograms, each with {data_z.shape[1]} time steps.")
else:
    print(f"Processed data not found in {processed_geo_dir}. Run the simulation notebook first.")

# %% [markdown]
# ## Convert Particle Velocity to DAS Strain Rate
# 
# Now we'll convert the particle velocity data to DAS strain-rate measurements using the DASConverter class. For this example, we'll focus on the X-component data, assuming the fiber is oriented along the X-axis.

# %%
# Initialize DAS converter
das_converter = DASConverter()

# Convert X-component data to DAS strain rate
das_data = das_converter.convert_numpy(
    data_x,  # X-component velocity data
    gauge_length=gauge_length,
    channel_spacing=channel_spacing,
    dt=times[1] - times[0]  # Time step
)

print(f"Converted {das_data.shape[0]} channels of DAS data, each with {das_data.shape[1]} time steps.")

# %%
# Visualize a single DAS channel
channel_idx = 50  # Middle channel
plot_seismic_trace(times, das_data[channel_idx], 
                  title=f"DAS Strain Rate - Channel {channel_idx+1}",
                  xlabel="Time (s)", ylabel="Strain Rate")

# %%
# Plot the DAS gather (all channels)
plot_seismic_gather(das_data, title="DAS Strain Rate Gather", 
                   xlabel="Time Sample", ylabel="Channel")

# %% [markdown]
# ## Compare Geophone and DAS Data
# 
# Let's compare the geophone (particle velocity) data with the derived DAS (strain rate) data to understand the relationship between these two measurements.

# %%
# Normalize data for comparison
def normalize(data):
    """Normalize data to [-1, 1] range."""
    for i in range(data.shape[0]):
        max_val = np.max(np.abs(data[i, :]))
        if max_val > 0:
            data[i, :] = data[i, :] / max_val
    return data

# Normalize both datasets
norm_geo_data = normalize(data_x.copy())
norm_das_data = normalize(das_data.copy())

# %%
# Compare geophone and DAS data for a single channel
channel_idx = 50  # Middle channel

plot_multiple_traces(
    times, 
    [norm_geo_data[channel_idx], norm_das_data[channel_idx]], 
    labels=["Geophone (Velocity)", "DAS (Strain Rate)"],
    title=f"Comparison of Geophone and DAS Data - Channel {channel_idx+1}",
    xlabel="Time (s)", 
    ylabel="Normalized Amplitude"
)

# %%
# Plot side-by-side comparison of the gathers
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot geophone data
im1 = ax1.imshow(norm_geo_data, aspect='auto', cmap='seismic')
ax1.set_title("Geophone (Velocity)")
ax1.set_xlabel("Time Sample")
ax1.set_ylabel("Channel")
plt.colorbar(im1, ax=ax1)

# Plot DAS data
im2 = ax2.imshow(norm_das_data, aspect='auto', cmap='seismic')
ax2.set_title("DAS (Strain Rate)")
ax2.set_xlabel("Time Sample")
plt.colorbar(im2, ax=ax2)

plt.suptitle("Comparison of Geophone and DAS Data")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analysis of Differences
# 
# Let's analyze the spectral differences between geophone and DAS data to better understand how they differ.

# %%
# Analyze frequency content
from scipy import signal

# Select a channel for analysis
channel_idx = 50

# Compute power spectral density for both geophone and DAS data
fs = 1.0 / (times[1] - times[0])  # Sampling frequency
f_geo, psd_geo = signal.welch(norm_geo_data[channel_idx], fs, nperseg=1024)
f_das, psd_das = signal.welch(norm_das_data[channel_idx], fs, nperseg=1024)

# Plot the spectra
plt.figure(figsize=(12, 6))
plt.semilogy(f_geo, psd_geo, label='Geophone')
plt.semilogy(f_das, psd_das, label='DAS')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (power/Hz)')
plt.title(f'Power Spectral Density - Channel {channel_idx+1}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Compute coherence between geophone and DAS data
f, coh = signal.coherence(norm_geo_data[channel_idx], norm_das_data[channel_idx], fs, nperseg=1024)

plt.figure(figsize=(12, 6))
plt.plot(f, coh)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.title(f'Coherence between Geophone and DAS - Channel {channel_idx+1}')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Impact of Gauge Length
# 
# Let's explore how changing the gauge length affects the DAS response.

# %%
# Try different gauge lengths
gauge_lengths = [5.0, 10.0, 20.0, 40.0]  # Gauge lengths in meters
das_multi_gauge = []

for gl in gauge_lengths:
    das_data_gl = das_converter.convert_numpy(
        data_x,
        gauge_length=gl,
        channel_spacing=channel_spacing,
        dt=times[1] - times[0]
    )
    # Normalize and store for the selected channel
    das_multi_gauge.append(normalize(das_data_gl.copy())[channel_idx])

# Plot comparison of different gauge lengths for a single channel
labels = [f"Gauge Length = {gl} m" for gl in gauge_lengths]
plot_multiple_traces(
    times, 
    das_multi_gauge, 
    labels=labels,
    title=f"DAS Response for Different Gauge Lengths - Channel {channel_idx+1}",
    xlabel="Time (s)", 
    ylabel="Normalized Amplitude"
)

# %% [markdown]
# ## Save DAS Data for Interpolation Model
# 
# Now that we have generated and analyzed the DAS data, let's save it for use in the interpolation model.

# %%
# Save DAS data
np.save(Path(processed_das_dir) / "das_data.npy", das_data)

# Save DAS metadata
das_metadata = {
    "gauge_length": gauge_length,
    "channel_spacing": channel_spacing,
    "n_channels": das_data.shape[0],
    "n_time_steps": das_data.shape[1],
    "dt": times[1] - times[0],
    "t_start": times[0],
    "t_end": times[-1]
}

# Convert to DataFrame and save as CSV
pd.DataFrame([das_metadata]).to_csv(Path(processed_das_dir) / "das_metadata.csv", index=False)

print(f"Saved DAS data to {processed_das_dir}")

# %% [markdown]
# ## Create Combined Dataset for Model Training
# 
# Finally, let's create a combined dataset containing both geophone and DAS data, aligned in space and time, for use in the interpolation model.

# %%
# Create a combined dataset
combined_data = {
    "times": times,
    "geophone_data": data_x,  # X-component only
    "das_data": das_data,
    "station_coordinates": station_df[["lat", "lon", "elevation"]].values,
    "metadata": {
        "gauge_length": gauge_length,
        "channel_spacing": channel_spacing,
        "dt": times[1] - times[0],
        "n_geophones": data_x.shape[0],
        "n_das_channels": das_data.shape[0]
    }
}

# Save the combined dataset
np.save(Path(processed_geo_dir) / "combined_dataset.npy", combined_data)
print(f"Saved combined dataset to {processed_geo_dir}")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we have:
# 
# 1. Loaded synthetic geophone data generated by SPECFEM3D
# 2. Converted particle velocity to DAS strain rate
# 3. Visualized and compared geophone and DAS measurements
# 4. Analyzed the spectral differences between the two data types
# 5. Explored the impact of different gauge lengths on DAS response
# 6. Saved the processed data for use in the interpolation model
# 
# The generated DAS data, alongside the synthetic geophone data, will now be used to train the multimodal interpolation model to fill in missing geophone channels.


