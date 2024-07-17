import numpy as np
import scipy as sp
import pyaudio
import wave
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import multiprocessing
import time
import scipy.io.wavfile
from pathlib import Path
from scipy.signal import chirp, spectrogram
import View_FFT_v2,Signal_generators


root_folder = Path(__file__).parents[2]
data_folder = root_folder / Path("Recordings/Trials/")
calibration_folder = root_folder / Path("Recordings/Reference/")
spectral_bins = np.array([1,100,300,500,750,1000,2000,3000,5000,7000,10000,12000,15000,20000,22050])
file_extension = ".wav"
freq_lower = 1000
freq_upper = 10000

#input_array = ['mahogany_nut_1_1','plywood_nut_1']
#output_array= ['mahogany_bridge_1_1','plywood_bridge_1']
input_array = ['mahogany_nut_1_1']
output_array= ['mahogany_bridge_1_1']

#input_array = ['old_exciter_sweep_1']
#output_array= ['new_exciter_sweep_1']
#input_array = ['headstock_strum_1']
#output_array= ['pickup_strum_1']


sampling_rate = 44100  # Hz
db_floor = -90  # dB


xscale = 'log'
yscale = 'linear'


if __name__ == "__main__":
    legend = 0
    #plot_fft(filepath, filepath2, ref_path, noise_path, sampling_rate)

    figure_count = 1
    #figure_count = plot_fft_array(input_array,sampling_rate,figure_count,False)
    #figure_count = plot_fft_power_array(array,sampling_rate,figure_count)
    figure_count = View_FFT_v2.plot_PSD(input_array,1024,figure_count,freq_lower,freq_upper,spectral_bins)
    figure_count = View_FFT_v2.plot_PSD(output_array,1024,figure_count,freq_lower,freq_upper,spectral_bins)
    figure_count = 2
    figure_count = View_FFT_v2.plot_PSD_db(input_array,1024,figure_count,freq_lower,freq_upper,spectral_bins)
    figure_count = View_FFT_v2.plot_PSD_db(output_array,1024,figure_count,freq_lower,freq_upper,spectral_bins)
    figure_count = 3
    figure_count = View_FFT_v2.plot_TF(input_array,output_array,legend,1024,figure_count,False,freq_lower,freq_upper,spectral_bins)
    #figure_count = plot_fft_dif_array(array,array2,sampling_rate,figure_count)

    plt.show()