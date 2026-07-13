"""
=========================================================
observations.py

Automatic AVAS Research Evaluation

Generates engineering observations from
FFT and psychoacoustic analysis.

Author:
COEP - ARAI Internship
=========================================================
"""

from dataclasses import dataclass

from Core.Pre_Compliance_Check.fft_analysis import FFTResult


# -------------------------------------------------------
# Observation Generator
# -------------------------------------------------------

class AVASObservations:

    def __init__(self):
        pass

    # ---------------------------------------------------

    def generate(self, fft: FFTResult):

        observations = []

        # -------------------------------------------------
        # Dominant Frequency
        # -------------------------------------------------

        f = fft.dominant_frequency

        observations.append(
            f"Dominant frequency detected at {f:.1f} Hz."
        )

        if 500 <= f <= 4000:

            observations.append(
                "Dominant frequency lies within the typical AVAS operating frequency range (approximately 500–4000 Hz)."
            )

        elif f < 500:

            observations.append(
                "Dominant frequency is relatively low and may reduce detectability in noisy urban environments."
            )

        else:

            observations.append(
                "Dominant frequency is relatively high and may increase perceived sharpness."
            )

        # -------------------------------------------------
        # UN R138 Frequency Region
        # -------------------------------------------------

        if 160 <= f <= 5000:

            observations.append(
                "The dominant frequency falls within the 160–5000 Hz frequency region considered by UN R138 octave-band analysis."
            )

        else:

            observations.append(
                "Dominant frequency lies outside the principal UN R138 octave-band region."
            )

        # -------------------------------------------------
        # Spectral Centroid
        # -------------------------------------------------

        centroid = fft.spectral_centroid

        observations.append(
            f"Spectral centroid = {centroid:.1f} Hz."
        )

        if centroid > 2000:

            observations.append(
                "High spectral centroid indicates a brighter sound character that may improve pedestrian detectability."
            )

        elif centroid > 1000:

            observations.append(
                "Moderate spectral centroid indicates balanced frequency content."
            )

        else:

            observations.append(
                "Low spectral centroid indicates dominance of low-frequency components."
            )

        # -------------------------------------------------
        # Bandwidth
        # -------------------------------------------------

        bw = fft.bandwidth

        observations.append(
            f"Spectral bandwidth = {bw:.1f} Hz."
        )

        if bw > 2500:

            observations.append(
                "Broad spectral bandwidth suggests broadband acoustic content."
            )

        elif bw > 1200:

            observations.append(
                "Moderate bandwidth indicates mixed tonal and broadband characteristics."
            )

        else:

            observations.append(
                "Narrow bandwidth suggests a predominantly tonal signal."
            )

        # -------------------------------------------------
        # Spectral Flatness
        # -------------------------------------------------

        flat = fft.flatness

        observations.append(
            f"Spectral flatness = {flat:.3f}."
        )

        if flat < 0.10:

            observations.append(
                "Low spectral flatness indicates a highly tonal signal."
            )

        elif flat < 0.30:

            observations.append(
                "Signal contains both tonal and broadband components."
            )

        else:

            observations.append(
                "High spectral flatness indicates a noise-like spectrum."
            )

        # -------------------------------------------------
        # Zero Crossing Rate
        # -------------------------------------------------

        zcr = fft.zero_crossing_rate

        observations.append(
            f"Zero-crossing rate = {zcr:.3f}."
        )

        if zcr > 0.10:

            observations.append(
                "Higher zero-crossing rate indicates stronger high-frequency activity."
            )

        else:

            observations.append(
                "Lower zero-crossing rate is consistent with tonal or harmonic content."
            )

        # -------------------------------------------------
        # Top Peaks
        # -------------------------------------------------

        observations.append(
            f"{len(fft.top_peaks)} dominant spectral peaks detected."
        )

        if len(fft.top_peaks) >= 4:

            observations.append(
                "Multiple prominent peaks suggest harmonic or multi-tone AVAS design."
            )

        elif len(fft.top_peaks) >= 2:

            observations.append(
                "More than one dominant peak indicates multiple tonal components."
            )

        else:

            observations.append(
                "Single dominant spectral component detected."
            )

        # -------------------------------------------------
        # Rolloff
        # -------------------------------------------------

        roll = fft.rolloff

        observations.append(
            f"95% spectral roll-off frequency = {roll:.1f} Hz."
        )

        if roll > 4000:

            observations.append(
                "Significant acoustic energy extends into higher frequencies."
            )

        else:

            observations.append(
                "Most acoustic energy is concentrated below 4 kHz."
            )

        # -------------------------------------------------
        # Research Conclusion
        # -------------------------------------------------

        observations.append("")

        observations.append("Overall AVAS Research Assessment")

        observations.append(
            "• Signal characteristics are consistent with an electronically generated warning sound."
        )

        observations.append(
            "• Frequency-domain analysis indicates that the signal contains prominent tonal components suitable for psychoacoustic evaluation."
        )

        observations.append(
            "• Frequency content falls within the analysis range recommended for AVAS evaluation."
        )

        observations.append(
            "• Absolute regulatory compliance cannot be determined because the recording has not been acoustically calibrated."
        )
                # -------------------------------------------------
        # Potential Improvements
        # -------------------------------------------------

        observations.append("")
        observations.append("Potential Improvements")

        if fft.flatness < 0.10:
            observations.append(
                "• Consider introducing broader spectral content if reduced tonality is desired."
            )

        if fft.spectral_centroid < 1000:
            observations.append(
                "• Increasing higher-frequency content may improve pedestrian detectability."
            )

        if len(fft.top_peaks) == 1:
            observations.append(
                "• Introducing additional harmonic components may improve sound richness."
            )
        return observations

    # ---------------------------------------------------

    def print_summary(self, fft):

        observations = self.generate(fft)

        print("\n")
        print("="*60)
        print("AUTOMATIC AVAS RESEARCH EVALUATION")
        print("="*60)

        for obs in observations:

            print("•", obs)

        print("="*60)

        return observations
