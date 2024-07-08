from scipy.signal import chirp
import scipy as sp
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

from pathlib import Path


root_folder = Path(__file__).parents[2]
data_folder = root_folder / Path("Recordings/Trials/")
calibration_folder = root_folder / Path("Recordings/Reference/")
spectral_bins = np.array([1,100,300,500,750,1000,2000,3000,5000,7000,10000,12000,15000,20000,22050])
file_extension = ".wav"

#input_array = ['mahogany_nut_1_1','plywood_nut_1']
#output_array= ['mahogany_bridge_1_1','plywood_bridge_1']
#input_array = ['mahogany_nut_1_1']
#output_array= ['mahogany_bridge_1_1']
input_array = ['headstock_strum_1']
output_array= ['pickup_strum_1']


sampling_rate = 44100  # Hz
db_floor = -90  # dB


xscale = 'log'
yscale = 'linear'

def generate_chirp(filename, sampling_rate, duration=5.0):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = chirp(t, f0=20, f1=sampling_rate / 2, t1=duration, method='linear')
    signal = signal * (2**15 - 1) / np.max(np.abs(signal))
    signal = signal.astype(np.int16)
    wav.write(filename, sampling_rate, signal)

generate_chirp("chirp.wav", sampling_rate)


def plot_fft_array(file_array, sampling_rate,fc,lims):
    max_amplitude_db = db_floor
    max_power_db = db_floor
    fc +=1

    # First pass to determine max values for y-axis limits
    for n in file_array:
        try:
            fft_out, psd, _ = fft_process(n, sampling_rate)
            max_amplitude_db = max(max_amplitude_db, max(fft_out))
            max_power_db = max(max_power_db, max(psd))
        except Exception as e:
            print(f"Error processing file {n}: {e}")

    # Second pass to plot the data
    for n in file_array:
        try:
            fft_out, psd, freqs = fft_process(n, sampling_rate)
            smoothed = sp.signal.savgol_filter(fft_out,1000,2)

            plt.figure(fc, figsize=(12, 6))
            plt.xscale(xscale)
            plt.plot(freqs, fft_out, label=n)
            plt.plot(freqs, smoothed, label=n+' (smoothed)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            plt.title('FFT Analysis')
            plt.legend()
            plt.xlim(1,sampling_rate/2)
            if lims:
                plt.ylim(db_floor, max_amplitude_db)

        except Exception as e:
            print(f"Error plotting FFT: {e}")

    return fc

def plot_fft_dif_array(file_array, file_array2, sampling_rate,fc):
    max_amplitude_db = db_floor
    max_power_db = db_floor


    # First pass to determine max values for y-axis limits
    for n in range(len(file_array)):
        fc +=1
        try:
            fft_out, psd, freqs = fft_process(file_array[n], sampling_rate)

            fft_out_sub,psd_sub,freqs_sub = fft_process(file_array2[n], sampling_rate)
            max_amplitude_db = max(max_amplitude_db, max(fft_out))
            max_power_db = max(max_power_db, max(psd))
            fm,f = fft_sub(freqs,fft_out,freqs_sub,fft_out_sub)
            print(fm)
            print(f)
    

            plt.figure(fc, figsize=(12, 6))
            plt.xscale(xscale)
            plt.plot(f, fm, label=n)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            plt.title('FFT Analysis')
            plt.legend()


        except Exception as e:
            print(f"Error plotting FFT: {e}")

    return fc

def plot_fft_power_array(file_array, sampling_rate,fc):
    fc +=1
    max_amplitude_db = db_floor
    max_power_db = db_floor

    # First pass to determine max values for y-axis limits
    for n in file_array:
        try:
            fft_out, psd, _ = fft_process(n, sampling_rate)
            max_amplitude_db = max(max_amplitude_db, max(fft_out))
            max_power_db = max(max_power_db, max(psd))
        except Exception as e:
            print(f"Error processing file {n}: {e}")

    # Second pass to plot the data
    for n in file_array:
        try:
            fft_out, psd, freqs = fft_process(n, sampling_rate)

            plt.figure(fc, figsize=(12, 6))
            plt.xscale(xscale)
            plt.plot(freqs, psd, label=n)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')
            plt.grid(True)
            plt.title('FFT Power Analysis')
            plt.legend()
            plt.ylim(db_floor, max_power_db)

        except Exception as e:
            print(f"Error plotting FFT: {e}")

    return fc

def plot_PSD(filearray,nperseg,fc):
    for n in filearray:
        sample_rate, data = open_file(n)

        # If the audio file has multiple channels (e.g., stereo), select one channel
        if data.ndim > 1:
            data = data[:, 0]

        # Calculate the Power Spectral Density (PSD) using Welch's method
        f, Pxx = sp.signal.welch(data, fs=sample_rate, nperseg=nperseg)
        

        # Plot the PSD
        plt.figure(fc,figsize=(10, 6))
        plt.semilogy(f, Pxx, label = n)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.xscale(xscale)
        plt.ylabel('PSD [V^2/Hz]')
        plt.legend()
        plt.grid(True)
    return fc

def plot_PSD_db(filearray,nperseg,fc):
    for n in filearray:
        sample_rate, data = open_file(n)

        # If the audio file has multiple channels (e.g., stereo), select one channel
        if data.ndim > 1:
            data = data[:, 0]

        # Calculate the Power Spectral Density (PSD) using Welch's method
        f, Pxx = sp.signal.welch(data, fs=sample_rate, nperseg=nperseg)
        Pxx = 10 * np.log10(Pxx)

        # Plot the PSD
        plt.figure(fc,figsize=(10, 6))
        plt.plot(f, Pxx, label = n)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.xscale(xscale)
        plt.ylabel('PSD [dB]')
        plt.legend()
        plt.grid(True)
    return fc

def plot_TF(filearray,filearray2,nperseg,fc,split):


    for n in range(len(filearray)):
        if split: 
            fc+=1
        f1 = filearray[n]
        f2 = filearray2[n]
        header = prefix_sort(f1,f2)
        sample_rate, input_data = open_file(f1)
        sample_rate, output_data = open_file(f2)

        # If the audio file has multiple channels (e.g., stereo), select one channel
        if input_data.ndim > 1:
            input_data = input_data[:, 0]
        if output_data.ndim > 1:
            output_data = output_data[:, 0]
        # Calculate the Power Spectral Density (PSD) using Welch's method
        f_in, Pxx_in = sp.signal.welch(input_data, fs=sample_rate, nperseg=nperseg)
        f_out, Pxx_out = sp.signal.welch(output_data, fs=sample_rate, nperseg=nperseg)

        TF = Pxx_out/Pxx_in
        TF_db = 10 * np.log10(TF)
        # Plot the PSD
        plt.figure(fc,figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.xscale(xscale)
        plt.plot(f_in, TF, label = header)
        plt.title(f'{header} Transfer Function (Linear Scale)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.legend()

        # Decibel scale
        plt.subplot(2, 1, 2)
        plt.xscale(xscale)
        plt.plot(f_in, TF_db, label = header)
        plt.title(f'{header} Transfer Function (dB Scale)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB]')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
    return fc

def open_file(file):
    path = file + file_extension
    print(path)
    try:
        rate, data = wav.read(data_folder / path)
    except:
        rate, data = wav.read(calibration_folder / path)
    return rate,data

def fft_process(filename, sampling_rate):
    rate,data = open_file(filename)
    # Use only one channel if the data is stereo
    if len(data.shape) > 1:
        data = data[:, 0]

    # Remove DC offset
    data = data - np.mean(data)

    # Apply windowing
    window = np.hamming(len(data))
    data_windowed = data * window

    # Compute rFFT
    fft_out = np.fft.rfft(data_windowed)

    # Compute the frequency axis
    freqs = np.fft.rfftfreq(len(data_windowed), 1.0 / sampling_rate)

    # Normalize FFT output
    fft_out = np.abs(fft_out) / np.sum(window)
    power_spectrum = fft_out ** 2  # Convert amplitude to power

    # Convert amplitude and power to dB scale
    amplitude_spectrum_db = 20 * np.log10(fft_out)
    power_spectrum_db = 10 * np.log10(power_spectrum)

    return amplitude_spectrum_db, power_spectrum_db, freqs



def fft_sub(input_f, input_m, ref_f, ref_m):
    input_f = np.array(input_f)
    input_m = np.array(input_m)
    ref_f = np.array(ref_f)
    ref_m = np.array(ref_m)
    f = ref_f
    if len(input_f) < len(ref_f):  # grab the shorter of the two data samples, in theory they should be identical.
        f = input_f

    fm = input_m[0:len(f)] - ref_m[0:len(f)]

    return fm, f

def spectral_analysis(input_f, input_m, ref_f, ref_m):
    input_f = np.array(input_f)
    input_m = np.array(input_m)
    ref_f = np.array(ref_f)
    ref_m = np.array(ref_m)
    f = ref_f
    if len(input_f) < len(ref_f):  # grab the shorter of the two data samples, in theory they should be identical.
        f = input_f
    for n in range(len(spectral_bins) - 1):
        start = spectral_bins[n]
        end = spectral_bins[n + 1]
        indices = np.where((f >= start) & (f < end))
        data = input_m[indices] - ref_m[indices]
        avg = np.average(data)
        max_val = np.max(data)
        min_val = np.min(data)
        print(f'range| {start}-{end}  Avg dB| {avg}  Max| {max_val}  min| {min_val}')


def prefix_sort(str1, str2):
    # Find the minimum length of the two strings
    min_len = min(len(str1), len(str2))
    
    # Iterate through the characters and find the longest common prefix
    for i in range(min_len):
        if str1[i] != str2[i]:
            return str1[:i]  # Return the prefix up to the first differing character
    
    return str1[:min_len]  # Return the entire string if one is a substring of the other

if __name__ == "__main__":
    #plot_fft(filepath, filepath2, ref_path, noise_path, sampling_rate)

    figure_count = 1
    #figure_count = plot_fft_array(input_array,sampling_rate,figure_count,False)
    #figure_count = plot_fft_power_array(array,sampling_rate,figure_count)
    figure_count = plot_PSD(input_array,1024,figure_count)
    figure_count = plot_PSD(output_array,1024,figure_count)
    figure_count = 2
    figure_count = plot_PSD_db(input_array,1024,figure_count)
    figure_count = plot_PSD_db(output_array,1024,figure_count)
    figure_count = 3
    figure_count = plot_TF(input_array,output_array,1024,figure_count,False)
    #figure_count = plot_fft_dif_array(array,array2,sampling_rate,figure_count)

    plt.show()
