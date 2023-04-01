#!/usr/bin/env python

import numpy as np
from scipy import signal
from scipy.io import wavfile
import wavio
import matplotlib.pyplot as plt
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
    
def filter20_20k(x, sr):
    '''
    filters everything outside of 20 - 20_000 Hz
    '''
    
    nyq = 0.5 * sr
    sos = signal.butter(5, [20.0 / nyq, 20_000.0 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, x)

def deconvolve(a, b, sr): # per mono file
    '''
    a is the input sweep signal, h the impulse response, and b the microphone-recorded signal. 
    We have a * h = b (convolution here!). 
    Let's take the discrete Fourier transform, we have fft(a) * fft(h) = fft(b), 
    then h = ifft(fft(b) / fft(a)).
    '''
    
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
    return h


class WavFromFile:
    def __init__(self, filename):
        self.load(filename)

    def load(self, filename):
        self.samplerate, data = wavfile.read(filename)

        # Normalize input to 32bit float
        if data.dtype == np.int8:
            data = np.float32(data / (2**(24-1)))
        elif data.dtype == np.int16:
            data = np.float32(data / (2**(16-1)))
        elif data.dtype == np.int32:
            data = np.float32(data / (2**(32-1)))

        self.nch = 1
        if data.ndim > 1:
            self.nch = data.shape[1]
        ch_data = np.empty((self.nch, data.shape[0]), dtype=np.float32)

        if self.nch > 1:
            for ch in range(self.nch):
                ch_data[ch] = data[:, ch]
        else:
            ch_data[0] = data
        self.data = ch_data
        
def writewav(filename, ch_data, samplerate, bitdepth):
    wavdata = np.column_stack([data for data in ch_data])
    wavio.write(filename, wavdata, samplerate, sampwidth=bitdepth//8)
    print(f'File "{filename}" written ({len(ch_data)} channels, {bitdepth} bit, {samplerate}Hz samplerate)')

def array_bounds(data, threshold):
    start = end = 0
    # Scan from start
    for i, value in enumerate(data):
        if abs(value) > threshold:
            start = i
            break
    # Scan from end
    for i, value in enumerate(data[::-1]):
        if abs(value) > threshold:
            end = len(data) - i
            break
    return start, end

def crop(ch_data, threshold):
    start = ch_data[0].size
    end = 0
    return_data = []
    for data in ch_data:
        _start, _end = array_bounds(data, threshold)
        if _start < start: start = _start
        if _end > end: end = _end
    for data in ch_data:
        return_data.append(data[start:end])
    return return_data

def limit(ch_data, option=None):
    return_data = []
    if option == 'clip':
        for data in ch_data:
            return_data.append(np.clip(data, -1, 1))
        return return_data
    elif option == 'normalize':
        max_value = 0
        for data in ch_data:
            _max_value = max(abs(data.min()), data.max())
            if _max_value > max_value: max_value = _max_value
        for data in ch_data:
            return_data.append(data / max_value)
        return return_data
    return ch_data

def display_audio(data, samplerate, color, title, duration=0):
    from IPython.display import Audio
    from IPython.core.display import display
    
    display_data = data[:int(duration*samplerate)]
    if not duration:
        duration = len(data) / samplerate
    size = min(display_data.size, int(duration*samplerate))
    time = np.linspace(0, duration, num=size)
    plt.figure(figsize=(18, 2))
    plt.title(title)
    plt.plot(time, display_data, color=color, linewidth=0.2)
    plt.xlabel('Seconds')
    plt.box(False)
    plt.show()
    display(Audio(data=data, rate=samplerate))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('sweepfile', help='filename of original sweep')
    parser.add_argument('recfile', help='filename of recorded sweep (mono or multichannel)')
    parser.add_argument('outfile', help='filename for extracted impulse response (channels identical to recfile)')
    parser.add_argument('--limit', choices=['normalize', 'clip'], help='Normalize or clip resulting amplitudes')
    parser.add_argument('--crop', metavar='<threshold>', default=0, type=float, help='Crop resulting samples below threshold at start and end')
    parser.add_argument('--bitdepth', metavar='<bitdepth>', default=24, type=int, help='Set bit depth for outfile (defaults to 24)')
    args = parser.parse_args()

    sweep = WavFromFile(args.sweepfile)
    recording = WavFromFile(args.recfile)
    ir = []
    for rec in recording.data:
        ir_channel = deconvolve(sweep.data[0], rec, sweep.samplerate)
        ir.append(ir_channel)
    wave = crop(limit(ir, args.limit), args.crop)
    writewav(args.outfile, wave, recording.samplerate, args.bitdepth)
    