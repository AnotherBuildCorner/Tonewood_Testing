import pyaudio    

audio = pyaudio.PyAudio()

    # Check number of available input devices
num_devices = audio.get_device_count()
print(f"Number of input devices: {num_devices}")

    # List all available input devices
for i in range(num_devices):
    device_info = audio.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']}")

