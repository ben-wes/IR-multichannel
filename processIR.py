# trying to make a sweep deconvolution script for ambisonic microphones (4 channels until the moment)
# building on https://gist.github.com/josephernest/
# and my own code, 
# enrique tomas
# September 2022-2023

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wave
import wavfile
import ffmpeg
from pydub import AudioSegment

# AUX functions 
def ratio(dB):
    return np.power(10, dB * 1.0 / 20)

def padarray(A, length, before=0):
    t = length - len(A) - before
    if t > 0:
        width = (before, t) if A.ndim == 1 else ([before, t], [0, 0])
        return np.pad(A, pad_width=width, mode='constant')
    else:
        width = (before, 0) if A.ndim == 1 else ([before, 0], [0, 0])
        return np.pad(A[:length - before], pad_width=width, mode='constant')

def filter20_20k(x, sr): # filters everything outside out 20 - 20000 Hz
    nyq = 0.5 * sr
    sos = signal.butter(5, [20.0 / nyq, 20000.0 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, x)

def normalize(data, bits):
    normfactor = 1.0
    if bits == 8 or bits == 16 or bits == 24:
        normfactor = 2 ** (bits-1)
    data = np.float32(data) * 1.0 / normfactor
    return data

# 24bit -> 32bit conversion from https://github.com/mgeier/python-audio/blob/master/audio-files/audio-files-with-wave.ipynb
def pcm24to32_bytearray(data, channels=1): 
    if len(data) % 3 != 0:
        raise ValueError('Size of data must be a multiple of 3 bytes')

    size = len(data) // 3
    
    # reserve memory (initialized with null bytes):
    temp = bytearray(size * 4)

    for i in range(size):
        newidx = i * 4 + 1
        oldidx = i * 3
        temp[newidx:newidx+3] = data[oldidx:oldidx+3]

    return np.frombuffer(temp, dtype='<i4').reshape(-1, channels)

def extract_wav_channel(input_wav, channel):
    '''
    Take Wave_read object as an input and get one of its channels
    '''
    input_wav.setpos(0)

    nch   = input_wav.getnchannels()
    depth = input_wav.getsampwidth()
    sdata = input_wav.readframes(input_wav.getnframes())
    if depth == 3:
        sdata = pcm24to32_bytearray(sdata, channels=nch)
        depth = 4

    # Extract channel data
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError(f'sample width {depth} not supported'.format(depth))
    data = np.fromstring(sdata, dtype=typ)
    ch_data = normalize(data[channel::nch], depth*8)

    # return channel data
    return ch_data
    
def process(SWEEPFILE, WAVFILE, OUTFILE):
    #sweep_visualisation(SWEEPFILE)
    
    wav   = wave.open(WAVFILE)
    nch   = wav.getnchannels()
    sr    = wav.getframerate()

    print(f'{nch} channels found')
    channels = []
    #load previous files and get sample rate, bitrate and data arrays "a" and "b"
    _ , sweep, _ = wavfile.read(SWEEPFILE, normalized=True)

    for channel in range(nch):
        recording = extract_wav_channel(wav, channel)
        filename = str(channel)+OUTFILE
        h, sr = calculate_IR(sweep, recording, sr, filename)
        visualise_IR(h, sr)
        channels.append(AudioSegment.from_wav(filename))
        
    AudioSegment.from_mono_audiosegments(*channels).export(OUTFILE, format="wav")
    print(f'Exported {len(channels)}-channel IR file {OUTFILE}')

def calculate_IR(SWEEP, RECORDING, sr, OUTFILE):  #per mono file

    a = SWEEP
    b = RECORDING
    
    # Deconvolution 
    # a is the input sweep signal, h the impulse response, and b the microphone-recorded signal. 
    # We have a * h = b (convolution here!). 
    # Let's take the discrete Fourier transform, we have fft(a) * fft(h) = fft(b), 
    # then h = ifft(fft(b) / fft(a)).

    a = padarray(a, sr*50, before=sr*10)
    b = padarray(b, sr*50, before=sr*10)
    h = np.zeros_like(b)

    b1 = filter20_20k(b, sr)

    ffta = np.fft.rfft(a)
    fftb = np.fft.rfft(b1)
    ffth = fftb / ffta
    
    h1 = np.fft.irfft(ffth)
    h1 = filter20_20k(h1, sr)

    h = h1
    h = h[:10 * sr]
    h *= ratio(dB=40)

    # write to file
    # wavfile.write(OUTFILE, sr, h, normalized=True, bitrate=16)
    # print(f'Exported mono IR file {OUTFILE}')
    
    return h, sr

def visualise_IR(hh,sr):    
    #VISUALIZE IR (mono)

    #we need to extract sample rate and number of samples to work with the file
    n_samples = hh.size

    #calculate duration of the file
    t_audio = n_samples/sr

    #Plot the sweep
    # 1. create an array for the x axis - time with the exact time of each sample
    times = np.linspace(0, n_samples/sr, num=round(n_samples)) #if 16 bits
    #print("size of time axis array:", times.size)
    #print("size of sweep axis array:", h.size)

    #plot
    plt.figure(figsize=(15, 5))
    plt.plot(times, hh)
    plt.title('IR')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio/2)
    plt.show()
    
def sweep_visualisation(SWEEPFILE):
    #load previous files and get sample rate, bitrate and data arrays "a" and "b"
    sr, a, br = wavfile.read(SWEEPFILE, normalized=True)
    
    #VISUALIZE ORIGINAL SWEEP (mono)

    #we need to extract sample rate and number of samples to work with the file
    n_samples = a.size

    #calculate duration of the file
    t_audio = n_samples/sr

    #Plot the sweep
    # 1. create an array for the x axis - time with the exact time of each sample
    times = np.linspace(0, n_samples/sr, num=round(n_samples)) #if 16 bits
    #times = np.linspace(0, n_samples/sample_freq, num=round(n_samples/2)) #num=round(n_samples/2)) if 32 bits
    print("size of time axis array:", times.size)
    print("size of sweep axis array:", a.size)

    #plot
    plt.figure(figsize=(15, 5))
    plt.plot(times, a)
    plt.title('Original Sweep')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.show()
    
    # trim the first 25 seconds
    samples = a[:int(sr*25)]
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(samples, Fs=sr)
    plt.show()  