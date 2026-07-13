"""
=========================================================
fft_analysis.py

Frequency-domain analysis for AVAS Research Evaluation

Features
--------
✓ FFT Spectrum
✓ Dominant Frequency
✓ Top Spectral Peaks
✓ Spectral Centroid
✓ Spectral Bandwidth
✓ Spectral Roll-off
✓ Spectral Flatness
✓ Zero Crossing Rate

=========================================================
"""

from dataclasses import dataclass

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks


# -------------------------------------------------------
# Result Object
# -------------------------------------------------------

@dataclass
class FFTResult:

    frequencies: np.ndarray
    magnitude: np.ndarray

    dominant_frequency: float

    top_peaks: list

    spectral_centroid: float

    bandwidth: float

    rolloff: float

    flatness: float

    zero_crossing_rate: float


# -------------------------------------------------------
# FFT Analysis
# -------------------------------------------------------

class FFTAnalysis:

    def __init__(self, sample_rate):

        self.sample_rate = sample_rate

    # ---------------------------------------------------

    def compute_fft(self, signal):

        """
        Compute FFT magnitude spectrum.
        """

        N = len(signal)

        fft = np.fft.rfft(signal)

        magnitude = np.abs(fft)

        frequency = np.fft.rfftfreq(

            N,

            d=1/self.sample_rate

        )

        return frequency, magnitude

    # ---------------------------------------------------

    def dominant_frequency(self, frequencies, magnitude):

        idx = np.argmax(magnitude)

        return float(frequencies[idx])

    # ---------------------------------------------------

    def top_peaks(

            self,

            frequencies,

            magnitude,

            number=5

    ):

        peaks, _ = find_peaks(

            magnitude,

            distance=100,

            prominence=np.max(magnitude)*0.05

        )

        if len(peaks)==0:

            return []

        peak_mag = magnitude[peaks]

        order = np.argsort(peak_mag)[::-1]

        order = order[:number]

        result=[]

        for i in order:

            result.append(

                (

                    float(frequencies[peaks[i]]),

                    float(peak_mag[i])

                )

            )

        return result

    # ---------------------------------------------------

    def spectral_features(self, signal):

        centroid = librosa.feature.spectral_centroid(

            y=signal,

            sr=self.sample_rate

        )[0].mean()

        bandwidth = librosa.feature.spectral_bandwidth(

            y=signal,

            sr=self.sample_rate

        )[0].mean()

        rolloff = librosa.feature.spectral_rolloff(

            y=signal,

            sr=self.sample_rate

        )[0].mean()

        flatness = librosa.feature.spectral_flatness(

            y=signal

        )[0].mean()

        zcr = librosa.feature.zero_crossing_rate(

            signal

        )[0].mean()

        return (

            centroid,

            bandwidth,

            rolloff,

            flatness,

            zcr

        )

    # ---------------------------------------------------

    def analyze(self, signal):

        freq, mag = self.compute_fft(signal)

        dom = self.dominant_frequency(

            freq,

            mag

        )

        peaks = self.top_peaks(

            freq,

            mag

        )

        (

            centroid,

            bandwidth,

            rolloff,

            flatness,

            zcr

        ) = self.spectral_features(

            signal

        )

        return FFTResult(

            frequencies=freq,

            magnitude=mag,

            dominant_frequency=dom,

            top_peaks=peaks,

            spectral_centroid=centroid,

            bandwidth=bandwidth,

            rolloff=rolloff,

            flatness=flatness,

            zero_crossing_rate=zcr

        )

    # ---------------------------------------------------

    def plot(self, result):

        plt.figure(figsize=(10,5))

        plt.plot(

            result.frequencies,

            result.magnitude

        )

        plt.title("FFT Spectrum")

        plt.xlabel("Frequency (Hz)")

        plt.ylabel("Magnitude")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    # ---------------------------------------------------

    def summary(self, result):

        print("\n========== FFT SUMMARY ==========\n")

        print(

            f"Dominant Frequency : "

            f"{result.dominant_frequency:.1f} Hz"

        )

        print(

            f"Spectral Centroid : "

            f"{result.spectral_centroid:.1f} Hz"

        )

        print(

            f"Bandwidth : "

            f"{result.bandwidth:.1f} Hz"

        )

        print(

            f"Roll-off : "

            f"{result.rolloff:.1f} Hz"

        )

        print(

            f"Spectral Flatness : "

            f"{result.flatness:.4f}"

        )

        print(

            f"Zero Crossing Rate : "

            f"{result.zero_crossing_rate:.4f}"

        )

        print("\nTop Peaks\n")

        for i,(f,m) in enumerate(result.top_peaks,1):

            print(

                f"{i}. "

                f"{f:.1f} Hz"

            )