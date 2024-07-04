import numpy as np
import pyaudio
import wave
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import multiprocessing
import time
import scipy.io.wavfile
from pathlib import Path

root_folder = Path(__file__).parents[2]
data_folder = root_folder / Path("Recordings/Trials/")
calibration_folder = root_folder / Path("Recordings/Reference/")
ch_sel = 1
start_freq = 20  # Hz
end_freq = 20000  # Hz
duration = 5  # seconds
sampling_rate = 44100  # Hz
repeat_count = 3  # Number of times to repeat the chirp
output_counter = 1
output_file = "zerogain_L"
output_filename = f'{output_file}_{output_counter}.wav'
reference_file = "direct_1.wav"
noise_file = "zerogain_L_1.wav"
output_filename = data_folder / output_filename

db_floor = -60 #dB

def generate_chirp(start_freq, end_freq, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    frequencies = np.linspace(start_freq, end_freq, len(t))
    signal = np.sin(2 * np.pi * frequencies * t)
    scipy.io.wavfile.write("chirp.wav", sampling_rate, signal)
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

def play_chirp(signal, sampling_rate, event, repeat_count=1):
    chunk = 1024
    format = pyaudio.paFloat32  # Use float32 for output
    channels = 1
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sampling_rate,
                        output=True)

    print(f"Playing chirp {repeat_count} times...")
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

def plot_fft(filename, reference_filename, noise_file, sampling_rate):
    try:
        fft_out, freqs = fft_process(filename, sampling_rate)
        fft_out2, freqs2 = fft_process(reference_filename, sampling_rate)
        nft, nff = fft_process(noise_file, sampling_rate)

        plt.figure(1, figsize=(12, 6))

        plt.plot(freqs2, fft_out2, label='Ref')
        plt.plot(freqs, fft_out, label='Response')
        plt.plot(nff, nft, label='Noise')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.title(output_file)
        plt.legend()
        plt.ylim(db_floor, max(max(fft_out), max(fft_out2), max(nft)))

        plt.show()

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

def plot_waveform(filename, sampling_rate):
    try:
        rate, data = wav.read(filename)
        time_axis = np.linspace(0, len(data) / rate, num=len(data))

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Time Domain Waveform')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error plotting waveform: {e}")

if __name__ == "__main__":
    # Parameters

    # Generate the frequency sweep (chirp)
    chirp_signal = generate_chirp(start_freq, end_freq, duration, sampling_rate)

    # Create multiprocessing event to synchronize playback and recording
    event = multiprocessing.Event()

    # Create processes for playing chirp and recording response
    play_process = multiprocessing.Process(target=play_chirp, args=(chirp_signal, sampling_rate, event, repeat_count))
    record_process = multiprocessing.Process(target=record_audio, args=(output_filename, duration * repeat_count, sampling_rate, event, ch_sel))  # Change 0 to 1 for the right channel

    # Start both processes
    play_process.start()
    record_process.start()

    # Wait for both processes to finish
    play_process.join()
    record_process.join()

    # Perform FFT analysis and plot the response
    plot_waveform(output_filename, sampling_rate)
    plot_fft(output_filename, calibration_folder / reference_file, calibration_folder / noise_file, sampling_rate)
