import pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048*2  # Increased chunk size for better frequency resolution
NPERSEG = 2048*2  # Increased segment length for better frequency resolution
FREQ_MIN = 1
FREQ_MAX = 2000
filename = "24_Classical"
sub_filename = "blank"