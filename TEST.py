# Note: If any graphs appear on the screen, they must be closed before the code continues.

import matplotlib.pyplot as plt
import numpy as np
import wave
import math
import contextlib
from scipy.io.wavfile import read

testFile = 'test.wav'           # Original wave file
outname = 'filteredfile.wav'    # Filtered wave file
cutOffFrequency = 2000          # Cut off frequency (adjust here to see different results).

# This function is used to edit wave files and allows them to be manipulated.
def interpret_wavfile(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):
    # This determines what type of format the audio file is in and how the data is structured
    if sample_width == 1:
        datatype = np.uint8     # unsigned char (1 byte)
    elif sample_width == 2:
        datatype = np.int16     # signed short (2 bytes)
    else:
        raise ValueError("Only 8 and 16 bit audio formats are supported.")

    # gets the audio channels from the file (mono, stereo, etc.)
    audioChannels = np.frombuffer(raw_bytes, dtype=datatype)

    # interleaving will split apart the stereo sound to two different tracks (left and right).
    if interleaved:
        # If interleaved, then this means that sample N of channel A is behind sample N of channel B in raw data
        audioChannels.shape = (n_frames, n_channels)
        audioChannels = audioChannels.T
    else:
        # If not interleaved, this means the samples from channel A occur first, then the samples from channel B are afterwards.
        audioChannels.shape = (n_channels, n_frames)

    return audioChannels    # returns the channel information

# Convolves the audio channel with the equation derived for the running mean filter
def moving_average(x, windowSize):
    return np.convolve(x, np.ones(windowSize)/windowSize, mode='valid')

with contextlib.closing(wave.open(testFile,'rb')) as spf:
    # obtains basic data from the wave file
    sampleRate = spf.getframerate()
    sampleWidth = spf.getsampwidth()
    numberChannels = spf.getnchannels()
    numberFrames = spf.getnframes()

    Fs, data = read("test.wav")     # reads data from wave file
    signal_fft = np.fft.fft(data)   # takes fast fourier transform of data

    # Note: If any graphs appear on the screen must be closed before the code continues. Threads required to continue execution.

    plt.plot(abs(signal_fft))
    plt.title("Frequency Domain - Original Signal")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude Spectrum")
    plt.ylim(0, 30000)
    plt.xlim(0, 20000)
    plt.show()

    # extracts raw audio from multi-channel wav file
    audioSignal = spf.readframes(numberFrames*numberChannels)   # reads the frames of the wave file in order to convert it into a format which is manipulable
    spf.close()                                                 # closes read file
    channels = interpret_wavfile(audioSignal, numberFrames, numberChannels, sampleWidth, True)  # calls function to break down wave file to allow manipulation for filters

    frequencyCutoff = (cutOffFrequency/sampleRate)                      # gets the frequency cut off values divided by the sampling rate of the wav file
    N = int(math.sqrt(0.196201 + frequencyCutoff**2)/frequencyCutoff)   # creates moving average filter


    # Convolves moving average filter with original audio file (utilizes fast fourier transform)
    filtered = moving_average(channels[0], N).astype(channels.dtype)

    # Note: Any graphs appear on the screen must be closed before the code continues.
    plt.plot(filtered)
    plt.title("Frequency Domain - Filtered Signal")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude (Arbitrary Unit)")
    plt.ylim(0, 30000)
    plt.xlim(0, 20000)
    plt.show()

    # opens wav file for writing so the new filtered audio can be saved to disk
    wav_file = wave.open(outname, "w")
    wav_file.setparams((1, sampleWidth, sampleRate, numberFrames, spf.getcomptype(), spf.getcompname()))
    wav_file.writeframes(filtered.tobytes('C'))
    wav_file.close()

    # Original audio file is included with this python script. Additionally, the filtered audio will also save in the same file. Be mindful of speaker volume.

