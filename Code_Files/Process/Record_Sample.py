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

output_file = "Strandberg"
transducer = "HB"

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

def record():
    input("press enter to record the nut position sample")
    
    #(start_freq, end_freq, duration,repeat_count, sampling_rate,output_filename,chirp_name,chirp_type,ch_sel):
    Signal_generators.play_and_record_chirp(20,20000,duration,3,sampling_rate,TDn_chirp_filename,"chirp.wav","logarithmic",ch_sel)
    Signal_generators.play_and_record_noise(3,2,3,sampling_rate,TDn_noise_filename,"white_noise.wav",ch_sel)
    input("press enter to record the bridge position sample")
    Signal_generators.play_and_record_chirp(20,20000,duration,3,sampling_rate,TDb_chirp_filename,"chirp.wav","logarithmic",ch_sel)
    Signal_generators.play_and_record_noise(3,2,3,sampling_rate,TDb_noise_filename,"white_noise.wav",ch_sel)

def set_counter():
    output_counter = input("set the sample number counter: ")
    TNN =  f'{transducer_nut_file}_noise_{output_counter}'
    TNC = f'{transducer_nut_file}_chirp_{output_counter}'
    TBN = f'{transducer_bridge_file}_noise_{output_counter}'
    TBC = f'{transducer_bridge_file}_chirp_{output_counter}'
    #p_filename = data_folder / f'{pickup_file}.wav'
    TDn_noise_filename =data_folder / f'{TNN}.wav'
    TDn_chirp_filename =data_folder / f'{TNC}.wav'
    TDb_noise_filename = data_folder / f'{TBN}.wav'
    TDb_chirp_filename = data_folder / f'{TBC}.wav'

    return TNN,TNC,TBN,TBC



if __name__ == "__main__":
    
    TNN,TNC,TBN,TBC=set_counter()
    TDn_noise_filename =data_folder / f'{TNN}.wav'
    TDn_chirp_filename =data_folder / f'{TNC}.wav'
    TDb_noise_filename = data_folder / f'{TBN}.wav'
    TDb_chirp_filename = data_folder / f'{TBC}.wav'
    record()
    array1 = [TNC,TBC]
    array2 = [TNN,TBN]
    tf_in = [TNC,TNN]
    tf_out =[TBC,TBN]
    legend = [f"Chirp",f'White Noise',]
    x = input("show graphs y/n")
    if x == "y":

        View_FFT_v2.plot_PSD_db(array1,1024,1,plot_bands[0],plot_bands[1],spectral_bins)
        View_FFT_v2.plot_PSD_db(array2,1024,2,plot_bands[0],plot_bands[1],spectral_bins)

        View_FFT_v2.plot_TF_db(tf_in,tf_out,legend,1024,3,True,plot_bands[0],plot_bands[1],spectral_bins)

        plt.show()