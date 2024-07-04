import numpy as np
import pyaudio
import wave
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import multiprocessing
import time
import scipy.io.wavfile

def generate_chirp(start_freq, end_freq, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    frequencies = np.linspace(start_freq, end_freq, len(t))
    signal = np.sin(2 * np.pi * frequencies * t)
    scipy.io.wavfile.write("chirp.wav", sampling_rate, signal)
    return signal

def record_audio(filename, duration, sampling_rate, event):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1  # for mono recording, adjust as needed
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
            frames.append(data)
        except Exception as e:
            print(f"Error recording: {e}")
            break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    if frames:
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(sampling_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
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

def plot_fft(filename, sampling_rate):
    try:
        rate, data = wav.read(filename)
        fft_out = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(fft_out), 1.0/sampling_rate)

        plt.figure(figsize=(12, 6))
        plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(fft_out[:len(freqs)//2])), label='Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.title('FFT Analysis')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error plotting FFT: {e}")

if __name__ == "__main__":
    # Parameters
    start_freq = 20  # Hz
    end_freq = 20000  # Hz
    duration = 5  # seconds
    sampling_rate = 44100  # Hz
    repeat_count = 3  # Number of times to repeat the chirp
    output_filename = 'response.wav'

    # Generate the frequency sweep (chirp)
    chirp_signal = generate_chirp(start_freq, end_freq, duration, sampling_rate)

    # Create multiprocessing event to synchronize playback and recording
    event = multiprocessing.Event()

    # Create processes for playing chirp and recording response
    play_process = multiprocessing.Process(target=play_chirp, args=(chirp_signal, sampling_rate, event, repeat_count))
    record_process = multiprocessing.Process(target=record_audio, args=(output_filename, duration, sampling_rate, event))

    # Start both processes
    play_process.start()
    record_process.start()

    # Wait for both processes to finish
    play_process.join()
    record_process.join()

    # Perform FFT analysis and plot the response
    plot_fft(output_filename, sampling_rate)
