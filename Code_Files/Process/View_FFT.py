import numpy as np
import wave
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import multiprocessing
import time
import os
from pathlib import Path


root_folder = Path(__file__).parents[2]
data_folder = root_folder / Path("Recordings/Trials/")
calibration_folder = root_folder / Path("Recordings/Reference/")
spectral_bins = np.array([1,100,300,500,750,1000,2000,3000,5000,7000,10000,12000,15000,20000,22050])
file_extension = ".wav"

output_f = 'pie_bridge_lowgain_1' 
output_f2 = 'pie_neck_lowgain_1'
calibration_f ='direct_1'
noise_f = 'pie_noise_lowgain_1'

output_file = output_f + file_extension
output_file2 = output_f2 + file_extension
calibration_file = calibration_f + file_extension
noise_file = noise_f + file_extension
sampling_rate = 44100  # Hz
db_floor = -60 # dB

def plot_fft(filename, filename2, reference_filename, noise_file, sampling_rate):
    try:
        #rate, data = wav.read(filename)
        #fft_out = np.fft.fft(data)
        #freqs = np.fft.fftfreq(len(fft_out), 1.0/sampling_rate)
        fft_out,freqs=fft_process(filename, sampling_rate)
        fft_out2,freqs2=fft_process(reference_filename, sampling_rate)
        fft_out3,freqs3=fft_process(filename2, sampling_rate)
        nft,nff = fft_process(noise_file,sampling_rate)
        fftsub , f3 = fft_sub(freqs,fft_out,freqs2,fft_out2)


#        fft_out -= fft_out2
#        freqs -= freqs2
        


        plt.figure(1,figsize=(12, 6))

        plt.plot(freqs2, fft_out2, label=calibration_f)
        plt.plot(freqs, fft_out, label=output_f)
        plt.plot(freqs3, fft_out3, label=output_f2)
        plt.plot(nff, nft, label=noise_f)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.title('FFT Analysis')
        plt.legend()
        plt.ylim(db_floor, max(max(fft_out), max(fft_out2), max(nft)))

        plt.figure(2,figsize=(12, 6))
        #plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(fft_out[:len(freqs)//2])), label='Response')
        plt.plot(f3, fftsub, label='Response')


        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.title('Transmission Difference')
        plt.legend()
        plt.ylim(db_floor, max(max(fft_out), max(fft_out2), max(nft)))

        plt.figure(3,figsize=(12, 6))
        #plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(fft_out[:len(freqs)//2])), label='Response')
        plt.plot(freqs, fft_out, label=output_f)
        plt.plot(freqs3, fft_out3, label=output_f2)   


        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.title('Comparison')
        plt.legend()
        plt.ylim(db_floor, max(max(fft_out), max(fft_out2), max(nft)))
        plt.show()


        spectral_analysis(freqs,fft_out,freqs2,fft_out2)


    except Exception as e:
        print(f"Error plotting FFT: {e}")


def fft_process(filename, sampling_rate):
    rate, data = wav.read(filename)

    # Remove DC offset
    data = data - np.mean(data)

    # Apply windowing
    window = np.hamming(len(data))
    data_windowed = data * window

    # Compute FFT
    fft_out = np.fft.fft(data_windowed)
    fft_out = fft_out[:len(fft_out) // 2]  # Take only the positive frequencies

    # Compute the frequency axis
    freqs = np.fft.fftfreq(len(data_windowed), 1.0 / sampling_rate)
    freqs = freqs[:len(freqs) // 2]

    # Normalize FFT output
    fft_out = np.abs(fft_out) / len(data_windowed)
    fft_out = 20 * np.log10(fft_out)

    return fft_out, freqs

def fft_sub(input_f,input_m,ref_f,ref_m):
    input_f = np.array(input_f)
    input_m = np.array(input_m)
    ref_f = np.array(ref_f)
    ref_m = np.array(ref_m)
    f = ref_f
    if len(input_f) < len(ref_f): #grab the shorter of the two data samples, in theory they should be identical.
          f = input_f

    fm = input_m[0:len(f)]-ref_m[0:len(f)]

    return fm,f
def spectral_analysis(input_f,input_m,ref_f,ref_m):
    input_f = np.array(input_f)
    input_m = np.array(input_m)
    ref_f = np.array(ref_f)
    ref_m = np.array(ref_m)
    f = ref_f
    if len(input_f) < len(ref_f): #grab the shorter of the two data samples, in theory they should be identical.
          f = input_f
    for n in range(len(spectral_bins)-1):
            start = spectral_bins[n]
            end = spectral_bins[n+1]
            indicies = np.where((f >= start)&(f < end))
            data = input_m[indicies]-ref_m[indicies]
            avg = np.average(data)
            max =np.max(data)
            min = np.min(data)
            print(f'range| {start}-{end}  Avg dB| {avg}  Max| {max}  min| {min}')

                  


if __name__ == "__main__":
        cur_path = os.path.dirname(__file__)
        filepath = data_folder / output_file
        filepath2 = data_folder / output_file2
        ref_path = calibration_folder / calibration_file
        noise_path = calibration_folder / noise_file
        plot_fft(filepath, filepath2, ref_path, noise_path, sampling_rate)

