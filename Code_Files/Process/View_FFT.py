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



output_filename = 'response.wav'
calibration_file = 'chirp.wav'
sampling_rate = 44100  # Hz
noise_floor = 60 # dB

def plot_fft(filename, reference_filename, sampling_rate):
    try:
        #rate, data = wav.read(filename)
        #fft_out = np.fft.fft(data)
        #freqs = np.fft.fftfreq(len(fft_out), 1.0/sampling_rate)
        fft_out,freqs=fft_process(filename, sampling_rate)
        fft_out2,freqs2=fft_process(reference_filename, sampling_rate)
        
        f1 = np.array(fft_out)
        f2 = np.array(fft_out2)
        length = min(len(f1),len(f2))

        fftsub = f1[0:length]-f2[0:length]





#        fft_out -= fft_out2
#        freqs -= freqs2
        


        plt.figure(1,figsize=(12, 6))
        #plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(fft_out[:len(freqs)//2])), label='Response')
        plt.plot(freqs, fft_out, label='Response')
        plt.plot(freqs2, fft_out2, label='Ref')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.title('FFT Analysis')
        plt.legend()

        plt.figure(2,figsize=(12, 6))
        #plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(fft_out[:len(freqs)//2])), label='Response')
        plt.plot(freqs2, fftsub, label='Response')


        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.title('FFT Analysis')
        plt.legend()
        plt.show()


        spectral_analysis(freqs,fft_out,freqs2,fft_out2)


    except Exception as e:
        print(f"Error plotting FFT: {e}")


def fft_process(filename, sampling_rate):
        rate, data = wav.read(filename)
        fft_out = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(fft_out), 1.0/sampling_rate)
        freq = freqs[:len(freqs)//2]
        fft_out = 20 * np.log10(np.abs(fft_out[:len(freqs)//2]))


     

        return fft_out, freq

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
        filepath = data_folder / output_filename
        ref_path = calibration_folder / calibration_file
        plot_fft(filepath, ref_path, sampling_rate)

