import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.animation import FuncAnimation
import Params
# Parameters
FORMAT = Params.FORMAT
CHANNELS = Params.CHANNELS
RATE = Params.RATE
CHUNK = Params.CHUNK  # Increased chunk size for better frequency resolution
NPERSEG = Params.NPERSEG  # Increased segment length for better frequency resolution
FREQ_MIN = Params.FREQ_MIN
FREQ_MAX = Params.FREQ_MAX

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2, 1)

# Frequency axis
freqs = np.fft.rfftfreq(NPERSEG, 1 / RATE)
# Filtered frequency range
freq_indices = np.where((freqs >= FREQ_MIN) & (freqs <= FREQ_MAX))[0]
freqs = freqs[freq_indices]

# Initialize plot data
y = np.zeros(len(freqs))
max_values = np.zeros(len(freqs))
min_values_dB = np.full(len(freqs), -100)  # Initialize min values for dB scale

# Line plots for linear scale
line1, = ax1.plot(freqs, y, label='Live Signal')
line2, = ax1.plot(freqs, y, label='Max Values')
ax1.set_xlim(FREQ_MIN, FREQ_MAX)
ax1.legend()
ax1.set_title("Power Spectral Density (Linear)")

# Line plots for dB scale
line3, = ax2.plot(freqs, y, label='Live Signal (dB)')
line4, = ax2.plot(freqs, y, label='Max Values (dB)')
ax2.set_xlim(FREQ_MIN, FREQ_MAX)
ax2.legend()
ax2.set_title("Power Spectral Density (dB)")

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

# Animation
ani = FuncAnimation(fig, update, interval=50)

plt.tight_layout()
plt.show()

# Close the stream
stream.stop_stream()
stream.close()
audio.terminate()
