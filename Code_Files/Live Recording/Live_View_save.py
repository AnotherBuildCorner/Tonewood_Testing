import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.animation import FuncAnimation
import os
import Params
# Parameters

# Initialize PyAudio

FORMAT = Params.FORMAT
CHANNELS = Params.CHANNELS
RATE = Params.RATE
CHUNK = Params.CHUNK  # Increased chunk size for better frequency resolution
NPERSEG = Params.NPERSEG  # Increased segment length for better frequency resolution
FREQ_MIN = Params.FREQ_MIN
FREQ_MAX = Params.FREQ_MAX
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Frequency axis
freqs = np.fft.rfftfreq(NPERSEG, 1 / RATE)
# Filtered frequency range
freq_indices = np.where((freqs >= FREQ_MIN) & (freqs <= FREQ_MAX))[0]
freqs = freqs[freq_indices]

# Initialize plot data
y = np.zeros(len(freqs))
max_values = np.zeros(len(freqs))
min_values_dB = np.full(len(freqs), -100)  # Initialize min values for dB scale

# Initialize plots
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, (ax3_lin, ax3_db) = plt.subplots(2, 1)  # Two subplots: linear and dB scale

# Line plots for linear scale (Window 1)
line1, = ax1.plot(freqs, y, label='Live Signal')
line2, = ax1.plot(freqs, y, label='Max Values')
ax1.set_xlim(FREQ_MIN, FREQ_MAX)
ax1.legend()
ax1.set_title("Power Spectral Density (Linear)")

# Line plots for dB scale (Window 2)
line3, = ax2.plot(freqs, y, label='Live Signal (dB)')
line4, = ax2.plot(freqs, y, label='Max Values (dB)')
ax2.set_xlim(FREQ_MIN, FREQ_MAX)
ax2.legend()
ax2.set_title("Power Spectral Density (dB)")

# Initialize saved data plots (Window 3)
saved_lines_lin = []
saved_lines_db = []
ax3_lin.set_xlim(FREQ_MIN, FREQ_MAX)
ax3_lin.set_title("Saved Max Values (Linear)")
ax3_db.set_xlim(FREQ_MIN, FREQ_MAX)
ax3_db.set_title("Saved Max Values (dB)")

# Global variable to keep track of saved files
save_count = 1
base_filename = Params.filename
sub_filename = Params.sub_filename

def update(frame):
    global max_values, min_values_dB

    # Read data from the stream
    data = stream.read(CHUNK)
    data_np = np.frombuffer(data, dtype=np.int16)
    
    # Compute the Power Spectral Density (PSD)
    f, Pxx = welch(data_np, RATE, nperseg=NPERSEG)
    
    # Filter the PSD for the desired frequency range
    Pxx = Pxx[freq_indices]

    # Update live signal line (linear scale)
    line1.set_ydata(Pxx)
    
    # Update maximum values line (linear scale)
    max_values = np.maximum(max_values, Pxx)
    line2.set_ydata(max_values)
    
    # Adjust y-axis limits for linear scale
    ax1.set_ylim(0, max(max_values) * 1.1)

    # Convert PSD to dB
    Pxx_dB = 10 * np.log10(Pxx)
    max_values_dB = 10 * np.log10(max_values)
    
    # Update live signal line (dB scale)
    line3.set_ydata(Pxx_dB)
    
    # Update maximum values line (dB scale)
    line4.set_ydata(max_values_dB)
    
    # Update minimum values for dB scale
    min_values_dB = np.minimum(min_values_dB, Pxx_dB)
    
    # Adjust y-axis limits for dB scale
    ax2.set_ylim(min(min_values_dB) - 10, max(max_values_dB) + 10)
    
    return line1, line2, line3, line4

def save_max_values(event):
    global save_count
    if event.key == 'x':
        filename = f"{base_filename}_{sub_filename}_{save_count}.txt"
        np.savetxt(filename, max_values)
        
        # Plot the saved data on the third window
        saved_line_lin, = ax3_lin.plot(freqs, max_values, label=f'Saved Max {save_count}')
        saved_line_db, = ax3_db.plot(freqs, 10 * np.log10(max_values), label=f'Saved Max {save_count}')
        saved_lines_lin.append(saved_line_lin)
        saved_lines_db.append(saved_line_db)
        
        ax3_lin.legend()
        ax3_lin.set_ylim(0, max(max_values) * 1.1)
        ax3_db.legend()
        ax3_db.set_ylim(min(10 * np.log10(max_values)) - 10, max(10 * np.log10(max_values)) + 10)
        
        fig3.canvas.draw()
        
        save_count += 1

def main():
# Animation
    ani1 = FuncAnimation(fig1, update, interval=50)
    ani2 = FuncAnimation(fig2, update, interval=50)

    # Connect the key press event to the save function
    fig1.canvas.mpl_connect('key_press_event', save_max_values)
    fig2.canvas.mpl_connect('key_press_event', save_max_values)

    plt.tight_layout()
    plt.show()

    # Close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

sub_filename = input("Input Sub Filename: ")
main()