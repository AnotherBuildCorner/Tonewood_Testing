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
output_file = "strandberg"

plot_bands = [1000,10000]

pickup_file = f'{output_file}_pickup_{output_counter}'
transducer_neck_file = f'{output_file}_TD_neck_{output_counter}'
transducer_body_file = f'{output_file}_TD_body_{output_counter}'
p_filename = data_folder / f'{pickup_file}.wav'
TDn_filename =data_folder / f'{transducer_neck_file}.wav'
TDb_filename = data_folder / f'{transducer_body_file}.wav'


db_floor = -60 #dB

def record():
    input("press enter to record the neck pickup")
    Signal_generators.record_only(p_filename,duration,sampling_rate,ch_sel)
    input("press enter to record the neck transducer")
    Signal_generators.record_only(TDn_filename,duration,sampling_rate,ch_sel)
    input("press enter to record the body transducer")
    Signal_generators.record_only(TDb_filename,duration,sampling_rate,ch_sel)
if __name__ == "__main__":
    #record()
    array1 = [pickup_file,transducer_neck_file,transducer_body_file]
    tf_in = [pickup_file,pickup_file,transducer_neck_file]
    tf_out =[transducer_neck_file,transducer_body_file,transducer_body_file]
    legend = [f"pickup>neck",f"pickup>body",f"neck>body",]


    View_FFT_v2.plot_PSD_db(array1,1024,1,plot_bands[0],plot_bands[1])

    View_FFT_v2.plot_TF(tf_in,tf_out,legend,1024,2,True,1000,10000)

    plt.show()