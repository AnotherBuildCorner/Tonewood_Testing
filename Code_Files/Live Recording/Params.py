import pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
mult = 2
CHUNK = 1024*mult # Increased chunk size for better frequency resolution
NPERSEG = 1024*mult  # Increased segment length for better frequency resolution
FREQ_MIN = 1
FREQ_MAX = 2000
filename = "24_Classical"
sub_filename = "blank"