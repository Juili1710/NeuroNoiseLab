"""
=========================================================
octave_analysis.py

1/3-Octave Band Analysis

Implements IEC 61260 octave-band filtering for
sound quality and AVAS pre-compliance analysis.

=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

from scipy.signal import butter
from scipy.signal import sosfiltfilt

from Core.Pre_Compliance_Check.utils import (
    rms,
    pressure_to_spl,
)

from Core.Pre_Compliance_Check.calibration import (
    Calibration,
    digital_to_pressure,
)

from Core.Pre_Compliance_Check.a_weighting import (
    apply_a_weighting,
)

# -------------------------------------------------------
# IEC Preferred 1/3-Octave Centre Frequencies
#
# UN R138 uses only 160–5000 Hz
# -------------------------------------------------------

IEC_CENTER_FREQUENCIES = np.array([

    160,

    200,

    250,

    315,

    400,

    500,

    630,

    800,

    1000,

    1250,

    1600,

    2000,

    2500,

    3150,

    4000,

    5000

], dtype=float)


# -------------------------------------------------------
# Results
# -------------------------------------------------------

@dataclass
class OctaveResult:

    frequency: float

    spl: float | None

    calibrated: bool

    status: str = "NOT CHECKED"


# -------------------------------------------------------
# Main Analysis Class
# -------------------------------------------------------

class OctaveAnalysis:
    """
    IEC 61260 1/3-octave analysis.

    Notes
    -----
    Filtering is performed using
    Butterworth band-pass filters.

    Future versions can replace this
    with true IEC digital octave filters
    without changing the public API.
    """

    def __init__(
            self,
            sample_rate: int,
            order: int = 4
    ):

        self.sample_rate = sample_rate

        self.order = order

        self.center_frequencies = IEC_CENTER_FREQUENCIES

        self.filters = self.create_filter_bank()

    # ---------------------------------------------------
    # Frequency Limits
    # ---------------------------------------------------

    @staticmethod
    def band_limits(
            centre_frequency: float
    ):
        """
        IEC 1/3-octave limits.

        f1 = fc / 2^(1/6)

        f2 = fc * 2^(1/6)
        """

        factor = 2 ** (1 / 6)

        lower = centre_frequency / factor

        upper = centre_frequency * factor

        return lower, upper

    # ---------------------------------------------------
    # Butterworth Filter
    # ---------------------------------------------------

    def design_filter(
            self,
            centre_frequency: float
    ):
        """
        Design Butterworth band-pass filter.

        Returns
        -------
        SOS coefficients.
        """

        low, high = self.band_limits(
            centre_frequency
        )

        nyquist = self.sample_rate / 2

        low /= nyquist

        high /= nyquist

        # Prevent instability
        low = max(low, 1e-6)

        high = min(high, 0.999)

        sos = butter(

            self.order,

            [low, high],

            btype="bandpass",

            output="sos"

        )

        return sos

    # ---------------------------------------------------
    # Filter Bank
    # ---------------------------------------------------

    def create_filter_bank(self):
        """
        Create one Butterworth filter
        for every octave band.

        Returns
        -------
        dict

            {frequency : sos}
        """

        bank = {}

        for fc in self.center_frequencies:

            bank[fc] = self.design_filter(fc)

        return bank

    # ---------------------------------------------------
    # Apply One Filter
    # ---------------------------------------------------

    def filter_band(
            self,
            signal: np.ndarray,
            centre_frequency: float
    ):
        """
        Filter signal through one
        octave band.
        """

        sos = self.filters[
            centre_frequency
        ]

        return sosfiltfilt(
            sos,
            signal
        )

###part 2 
    # ---------------------------------------------------
    # Band SPL
    # ---------------------------------------------------

    def band_spl(
            self,
            signal: np.ndarray,
            centre_frequency: float,
            calibration: Calibration
    ) -> float | None:
        """
        Compute SPL for a single
        1/3-octave band.

        Parameters
        ----------
        signal
            Input audio signal.

        centre_frequency
            IEC centre frequency.

        calibration
            Calibration object.

        Returns
        -------
        float | None
        """

        # ----------------------------
        # Relative recordings
        # ----------------------------

        if not calibration.calibrated:

            return None

        # ----------------------------
        # Apply band filter
        # ----------------------------

        filtered = self.filter_band(

            signal,

            centre_frequency

        )

        # ----------------------------
        # Convert to pressure
        # ----------------------------

        pressure = digital_to_pressure(

            filtered,

            calibration

        )

        # ----------------------------
        # Band RMS
        # ----------------------------

        pressure_rms = rms(

            pressure

        )

        # ----------------------------
        # SPL
        # ----------------------------

        spl = pressure_to_spl(

            pressure_rms

        )

        return float(spl)

    # ---------------------------------------------------
    # Analyze
    # ---------------------------------------------------

    def analyze(
            self,
            signal: np.ndarray,
            calibration: Calibration,
            apply_weighting: bool = True
    ):
        """
        Compute complete
        1/3-octave spectrum.

        Parameters
        ----------
        signal

        calibration

        apply_weighting

            Apply IEC A-weighting
            before octave analysis.

        Returns
        -------
        List[OctaveResult]
        """

        # ----------------------------------
        # A-weighting
        # ----------------------------------

        if apply_weighting:

            signal = apply_a_weighting(

                signal,

                self.sample_rate

            )

        results = []

        # ----------------------------------
        # Loop over bands
        # ----------------------------------

        for fc in self.center_frequencies:

            value = self.band_spl(

                signal,

                fc,

                calibration

            )

            results.append(

                OctaveResult(

                    frequency=fc,

                    spl=value,

                    calibrated=calibration.calibrated

                )

            )

        return results

    # ---------------------------------------------------
    # Relative Analysis
    # ---------------------------------------------------

    def analyze_relative(
            
            self,
            signal: np.ndarray
    ):
        """
        Used when calibration
        is unavailable.

        Returns relative dB.

        Useful for
        research datasets.
        """

        results = []

        for fc in self.center_frequencies:

            filtered = self.filter_band(

                signal,

                fc

            )

            value = 20 * np.log10(

                rms(filtered)

                +

                1e-12

            )

            results.append(

                OctaveResult(

                    frequency=fc,

                    spl=float(value),

                    calibrated=False,

                    status="RELATIVE"

                )

            )

        return results
# Part 3 
    # ---------------------------------------------------
    # Compare with Standard
    # ---------------------------------------------------

    def compare_with_standard(
            self,
            results,
            required_levels
    ):
        """
        Compare measured octave levels
        with regulatory limits.

        Parameters
        ----------
        results
            List[OctaveResult]

        required_levels
            List of minimum SPL values.

        Returns
        -------
        List[OctaveResult]
        """

        if len(results) != len(required_levels):

            raise ValueError(
                "Required SPL table length mismatch."
            )

        for result, limit in zip(results, required_levels):

            if result.spl is None:

                result.status = "NOT CALIBRATED"

            elif result.spl >= limit:

                result.status = "PASS"

            else:

                result.status = "FAIL"

        return results

    # ---------------------------------------------------
    # DataFrame
    # ---------------------------------------------------

    def dataframe(
            self,
            results,
            required_levels=None
    ):
        """
        Convert octave results
        to pandas DataFrame.
        """

        rows = []

        for i, r in enumerate(results):

            row = {

                "Frequency (Hz)": r.frequency,

                "Measured SPL (dB)": r.spl,

                "Status": r.status

            }

            if required_levels is not None:

                row["Required SPL (dB)"] = required_levels[i]

                if r.spl is None:

                    row["Difference (dB)"] = None

                else:

                    row["Difference (dB)"] = (
                        r.spl - required_levels[i]
                    )

            rows.append(row)

        return pd.DataFrame(rows)

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------

    def plot(
            self,
            results,
            required_levels=None
    ):
        """
        Plot octave spectrum.
        """

        import matplotlib.pyplot as plt

        freq = [r.frequency for r in results]

        values = [

            np.nan if r.spl is None else r.spl

            for r in results

        ]

        plt.figure(figsize=(10,5))

        plt.semilogx(

            freq,

            values,

            marker="o",

            linewidth=2,

            label="Measured"

        )

        if required_levels is not None:

            plt.semilogx(

                freq,

                required_levels,

                "--",

                linewidth=2,

                label="UN R138 Minimum"

            )

        plt.xticks(freq, labels=[str(int(i)) for i in freq])

        plt.grid(True, which="both")

        plt.xlabel("Centre Frequency (Hz)")

        plt.ylabel("Sound Pressure Level (dB)")

        plt.title("1/3-Octave Band Spectrum")

        plt.legend()

        plt.tight_layout()

        plt.show()

    # ---------------------------------------------------
    # Export CSV
    # ---------------------------------------------------

    def export_csv(
            self,
            dataframe,
            filename
    ):
        """
        Export results to CSV.
        """

        dataframe.to_csv(

            filename,

            index=False

        )

        return filename

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------

    def summary(
            self,
            results
    ):
        """
        Return quick statistics.
        """

        valid = [

            r.spl

            for r in results

            if r.spl is not None

        ]

        if len(valid) == 0:

            return {

                "Calibrated": False,

                "Bands": len(results)

            }

        return {

            "Calibrated": True,

            "Bands": len(results),

            "Maximum SPL": np.max(valid),

            "Minimum SPL": np.min(valid),

            "Mean SPL": np.mean(valid)

        }