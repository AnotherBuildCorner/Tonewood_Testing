from scipy.signal import chirp, spectrogram
import numpy as np
import scipy.io.wavfile as wav

sampling_rate = 44100

def generate_chirp(filename, sampling_rate, duration=5.0):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = chirp(t, f0=20, f1=sampling_rate / 2, t1=duration, method='linear')
    signal = signal * (2**15 - 1) / np.max(np.abs(signal))
    signal = signal.astype(np.int16)
    wav.write(filename, sampling_rate, signal)

generate_chirp("chirp.wav", sampling_rate)