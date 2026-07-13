import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_spectrogram(audio, sr):

    D = librosa.amplitude_to_db(
        abs(librosa.stft(audio)),
        ref=np.max
    )

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        D,
        sr=sr,
        x_axis='time',
        y_axis='log'
    )

    plt.colorbar(format='%+2.0f dB')

    plt.title('Spectrogram')

    plt.tight_layout()

    plt.show(block=False)

    plt.pause(2)