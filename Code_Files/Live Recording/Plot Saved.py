import os
import numpy as np
import matplotlib.pyplot as plt
import Params
# Folder containing the .txt files (relative path)
folder_path = os.path.join(os.path.dirname(__file__), '2024-Classical')  # Adjust 'data_folder' to the correct folder name

FREQ_MIN = Params.FREQ_MIN
FREQ_MAX = Params.FREQ_MAX

# Sampling parameters
sampling_rate = Params.RATE  # Adjust according to your sampling rate
segment_length = Params.NPERSEG  # Adjust according to your segment length

# Generate the full frequency axis
full_freqs = np.fft.rfftfreq(segment_length, 1 / sampling_rate)

# Initialize plots
fig, (ax_lin, ax_db) = plt.subplots(2, 1, figsize=(10, 8))

# Loop through all .txt files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        # Full path to the file
        file_path = os.path.join(folder_path, filename)
        
        # Read the data from the file
        try:
            data = np.loadtxt(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        # Determine the frequency range from the file
        num_points = len(data)
        if num_points > len(full_freqs):
            print(f"Unexpected data length in {file_path}: {num_points} (Expected: {len(full_freqs)})")
            continue
        
        # Generate frequency axis based on the length of the data
        freqs = full_freqs[:num_points]
        
        # Filter frequency range for plotting
        freq_indices = np.where((freqs >= FREQ_MIN) & (freqs <= FREQ_MAX))[0]
        freqs_filtered = freqs[freq_indices]
        data_filtered = data[freq_indices]
        
        # Plot linear PSD
        ax_lin.plot(freqs_filtered, data_filtered, label=filename[:-4])  # Remove .txt extension for the legend
        
        # Plot dB PSD
        data_dB = 10 * np.log10(data_filtered + np.finfo(float).eps)  # Add epsilon to avoid log(0)
        ax_db.plot(freqs_filtered, data_dB, label=filename[:-4])  # Remove .txt extension for the legend

# Set titles and legends
ax_lin.set_title('Power Spectral Density (Linear)')
ax_lin.set_xlabel('Frequency (Hz)')
ax_lin.set_ylabel('Power')
ax_lin.legend()

ax_db.set_title('Power Spectral Density (dB)')
ax_db.set_xlabel('Frequency (Hz)')
ax_db.set_ylabel('Power (dB)')
ax_db.legend()

plt.tight_layout()
plt.show()
