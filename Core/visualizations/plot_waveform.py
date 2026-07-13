import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(audio, sr):

    time = np.linspace(
        0,
        len(audio) / sr,
        num=len(audio)
    )

    plt.figure(figsize=(12, 4))

    plt.plot(time, audio)

    plt.title("Waveform")

    plt.xlabel("Time (s)")

    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.tight_layout()

    #plt.show()
    plt.show(block=False)
plt.pause(0.1)