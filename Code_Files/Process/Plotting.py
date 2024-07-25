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
ch_sel = 1
sampling_rate = 44100  # Hz
duration = 5  # seconds


output_counter = 1

output_file_array = ["PurpleHeart","RedHeart"]
output_range_array = [1,1]

output_file = "o"
output_range = 1
transducer = "US"

plot_bands = [100,8000]

#pickup_file = f'{output_file}_pickup_{output_counter}'
transducer_nut_file = f'{output_file}_{transducer}_neck'
transducer_bridge_file = f'{output_file}_{transducer}_bridge'
TNN =  f'{transducer_nut_file}_noise_{output_counter}'
TNC = f'{transducer_nut_file}_chirp_{output_counter}'
TBN = f'{transducer_bridge_file}_noise_{output_counter}'
TBC = f'{transducer_bridge_file}_chirp_{output_counter}'
#p_filename = data_folder / f'{pickup_file}.wav'
TDn_noise_filename =data_folder / f'{TNN}.wav'
TDn_chirp_filename =data_folder / f'{TNC}.wav'
TDb_noise_filename = data_folder / f'{TBN}.wav'
TDb_chirp_filename = data_folder / f'{TBC}.wav'

spectral_bins = np.array([1,100,300,500,750,1000,2000,3000,5000,7000,10000,12000,15000,20000,22050])
db_floor = -60 #dB




def set_counter(output_file,output_counter):
    transducer_nut_file = f'{output_file}_{transducer}_neck'
    transducer_bridge_file = f'{output_file}_{transducer}_bridge'

    TNN =  f'{transducer_nut_file}_noise_{output_counter}'
    TNC = f'{transducer_nut_file}_chirp_{output_counter}'
    TBN = f'{transducer_bridge_file}_noise_{output_counter}'
    TBC = f'{transducer_bridge_file}_chirp_{output_counter}'
    #p_filename = data_folder / f'{pickup_file}.wav'
    TDn_noise_filename =data_folder / f'{TNN}.wav'
    TDn_chirp_filename =data_folder / f'{TNC}.wav'
    TDb_noise_filename = data_folder / f'{TBN}.wav'
    TDb_chirp_filename = data_folder / f'{TBC}.wav'
    return TNC,TNN,TBC,TBN

if __name__ == "__main__":
    for file in range(len(output_file_array)):
        o_file = output_file_array[file]

        o_range = output_range_array[file]

        for x in range(o_range):
            TNC,TNN,TBC,TBN = set_counter(o_file,x+1)
            array1 = [TNC,TBC]
            array2 = [TNN,TBN]
            tf_in = [TNC,TNN]
            tf_out =[TBC,TBN]
            legend = [f"Chirp",f'White Noise',]
            View_FFT_v2.plot_PSD_db(array1,1024,1,plot_bands[0],plot_bands[1],spectral_bins)
            View_FFT_v2.plot_PSD_db(array2,1024,2,plot_bands[0],plot_bands[1],spectral_bins)
            View_FFT_v2.plot_TF_db(tf_in,tf_out,legend,1024,3,True,plot_bands[0],plot_bands[1],spectral_bins)

    plt.show()