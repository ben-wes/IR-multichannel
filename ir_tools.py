import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from IPython.display import Audio 
from IPython.core.display import display
import warnings
warnings.filterwarnings('ignore')

def padarray(A, length, before=0):
    t = length - len(A) - before
    if t > 0:
        width = (before, t) if A.ndim == 1 else ([before, t], [0, 0])
        return np.pad(A, pad_width=width, mode='constant')
    else:
        width = (before, 0) if A.ndim == 1 else ([before, 0], [0, 0])
        return np.pad(A[:length - before], pad_width=width, mode='constant')
    
def filter20_20k(x, sr): # filters everything outside of 20 - 20_000 Hz
    nyq = 0.5 * sr
    sos = signal.butter(5, [20.0 / nyq, 20_000.0 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, x)

def ratio(dB):
    return np.power(10, dB * 1.0 / 20)

def deconvolve(a, b, sr): # per mono file
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

    h = h1[:10 * sr]
    h *= ratio(dB=40)
    return h

def readwav(filename):
    samplerate, data = wavfile.read(filename)
    data_type = data.dtype
    if data_type == np.int8:
        data = np.float32(data / (2**(24-1)))
    if data_type == np.int16:
        data = np.float32(data / (2**(16-1)))
    if data_type == np.int32:
        data = np.float32(data / (2**(32-1)))

    nch = 1
    if data.ndim > 1:
        nch = data.shape[1]
    ch_data = []
    if nch > 1:
        for ch in range(nch):
            ch_data.append(data[:, ch])
    else:
        ch_data.append(data)
    return samplerate, ch_data

def writewav(filename, ch_data, samplerate):
    wavdata = np.column_stack([data for data in ch_data])
    wavfile.write(filename, samplerate, wavdata)
    print(f'{len(ch_data)}-channel audio written to {filename}')
    
def display_audio(data, samplerate, color, title, duration=0):
    if not duration:
        duration = len(data) / samplerate
    time = np.linspace(0, duration, num=int(duration*samplerate))
    plt.figure(figsize=(18, 2))
    plt.title(title)
    plt.plot(time, data[:int(duration*samplerate)], color=color, linewidth=0.2)
    plt.xlabel('Time (s)')
    plt.box(False)
    plt.show()
    display(Audio(data, rate=samplerate))