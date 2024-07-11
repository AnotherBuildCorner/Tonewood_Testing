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
import View_FFT_v2

'''
features to add.
a sweep or white noise impulse decay test for sustain
search for peak transmission, and correlate with frequency
a stepped response using a series of guitar notes  say A4-ect
add a narrowband test with a longer sweep for guitar ranges


'''
root_folder = Path(__file__).parents[2]
data_folder = root_folder / Path("Recordings/Trials/")
calibration_folder = root_folder / Path("Recordings/Reference/")
ch_sel = 1
sampling_rate = 44100  # Hz
#sweep parameters
start_freq = 20  # Hz
end_freq = 20000  # Hz
sweep_duration = 5  # seconds

repeat_count = 3  # Number of times to repeat the chirp
chirp_type = 'logarithmic'
chirp_name = "chirp.wav"
#white noise parameters
noise_duration = 5 #total signal length
noise_silence = 2 #last x seconds of total length are silence
noise_repeats = 3 #number of repeats
noise_name = 'white_noise.wav'

#pulse parameters
impulse_time = 0
impulse_duration = 2
impulse_repeats = 2
impulse_name = "impulse.wav"
#file oputput parameters
output_counter = 2
output_file = "testing"


sweep_file = f'{output_file}_sweep_{output_counter}'
noise_file = f'{output_file}_noise_{output_counter}'
pulse_file = f'{output_file}_impulse_{output_counter}'
sweep_filename = data_folder / f'{sweep_file}.wav'
impulse_filename =data_folder / f'{pulse_file}.wav'
noise_filename = data_folder / f'{noise_file}.wav'


db_floor = -60 #dB

def generate_chirp(start_freq, end_freq, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    frequencies = np.linspace(start_freq, end_freq, len(t))
    signal = np.sin(2 * np.pi * frequencies * t)
    signal = np.float32(signal)
    scipy.io.wavfile.write("chirp.wav", sampling_rate, signal)
    
    return signal

def generate_pulse(duration,pulsetime,sampling_rate,filename):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    impulse = np.zeros(int(sampling_rate * duration))
    impulse[sampling_rate*pulsetime] = 1  # Dirac delta function
    signal2 = impulse * (2**15 - 1) / np.max(np.abs(impulse))
    signal2 = impulse.astype(np.int16)
    wav.write(filename, sampling_rate, signal2) 
    return impulse

def generate_white_noise(duration,post_silence,fs,filename):
    noise = np.random.normal(0, 1, int(fs * (duration-post_silence)))
    silence= np.zeros(int(fs*post_silence))
    signal = np.concatenate((noise,silence))
    signal2 = signal * (2**15 - 1) / np.max(np.abs(signal))
    signal2 = signal.astype(np.int16)
    wav.write(filename, sampling_rate, signal2)
    return signal
                    

def generate_chirp_linear(start_freq, end_freq, duration, sampling_rate,filename,met):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = chirp(t, f0=start_freq, f1=end_freq / 2, t1=duration, method=met)
    signal2 = signal * (2**15 - 1) / np.max(np.abs(signal))
    signal2 = signal.astype(np.int16)
    wav.write(filename, sampling_rate, signal2)

    return signal



def record_audio(filename, duration, sampling_rate, event, channel=0):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 2  # for stereo recording
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sampling_rate,
                        input=True,
                        frames_per_buffer=chunk)

    frames = []

    print("Recording...")
    start_time = time.time()
    while time.time() - start_time < duration and not event.is_set():
        try:
            data = stream.read(chunk)
            # Convert data to numpy array and extract the desired channel
            data = np.frombuffer(data, dtype=np.int16)
            data = data[channel::channels]  # Extract the specified channel
            frames.append(data.tobytes())
        except Exception as e:
            print(f"Error recording: {e}")
            break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    if frames:
        try:
            output_path = Path(filename)

            # Ensure the directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(1)  # mono output
                wf.setsampwidth(audio.get_sample_size(format))
                wf.setframerate(sampling_rate)
                wf.writeframes(b''.join(frames))
        except Exception as e:
            print(f"Error writing WAV file: {e}")

def play_signal(signal, sampling_rate, event, repeat_count=1):
    chunk = 1024
    format = pyaudio.paFloat32  # Use float32 for output
    channels = 1
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sampling_rate,
                        output=True)

    print(f"Playing signal {repeat_count} times...")
    try:
        for _ in range(repeat_count):
            stream.write(signal.astype(np.float32).tobytes())
            if event.is_set():
                break  # Exit loop if event is set (recording started)
    except Exception as e:
        print(f"Error playing chirp: {e}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Set event to notify recording function to stop
    event.set()


def play_and_record_chirp(start_freq, end_freq, duration,repeat_count, sampling_rate,output_filename,chirp_name,chirp_type,ch_sel):
    chirp_signal = generate_chirp_linear(start_freq, end_freq, duration, sampling_rate,chirp_name,chirp_type)
    event = multiprocessing.Event()
    play_process = multiprocessing.Process(target=play_signal, args=(chirp_signal, sampling_rate, event, repeat_count))
    record_process = multiprocessing.Process(target=record_audio, args=(output_filename, duration * repeat_count, sampling_rate, event, ch_sel))  # Change 0 to 1 for the right channel
    play_process.start()
    record_process.start()
    play_process.join()
    record_process.join()

def play_and_record_impulse(pulse_time,duration,repeat_count, sampling_rate,output_filename,pulse_name,ch_sel):
    pulse_signal = generate_pulse(duration,pulse_time, sampling_rate,pulse_name)
    event = multiprocessing.Event()
    play_process = multiprocessing.Process(target=play_signal, args=(pulse_signal, sampling_rate, event, repeat_count))
    record_process = multiprocessing.Process(target=record_audio, args=(output_filename, duration * repeat_count, sampling_rate, event, ch_sel)) 
    play_process.start()
    record_process.start()
    play_process.join()
    record_process.join()

def play_and_record_noise(duration,silence_time,repeat_count, sampling_rate,output_filename,noise_name,ch_sel):
    pulse_signal = generate_white_noise(duration,silence_time, sampling_rate,noise_name)
    event = multiprocessing.Event()
    play_process = multiprocessing.Process(target=play_signal, args=(pulse_signal, sampling_rate, event, repeat_count))
    record_process = multiprocessing.Process(target=record_audio, args=(output_filename, duration * repeat_count, sampling_rate, event, ch_sel)) 
    play_process.start()
    record_process.start()
    play_process.join()
    record_process.join()



if __name__ == "__main__":
    
    #play_and_record_chirp(start_freq, end_freq, sweep_duration,repeat_count, sampling_rate,sweep_filename,chirp_name,chirp_type,ch_sel)
    #play_and_record_impulse(impulse_time,impulse_duration,impulse_repeats, sampling_rate,impulse_filename,impulse_name,ch_sel)
    play_and_record_noise(noise_duration,noise_silence,noise_repeats, sampling_rate,noise_filename,noise_name,ch_sel)


    View_FFT_v2.plot_PSD([noise_file],1024,1)
    plt.show()
