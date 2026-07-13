"""
=========================================================
utils.py

General DSP utilities used throughout the
Sound Quality Framework.

This module contains reusable mathematical
functions shared by all compliance, feature
extraction and visualization modules.

Author:
COEP - ARAI Internship
=========================================================
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window
from typing import Tuple

# --------------------------------------------------------
# Constants
# --------------------------------------------------------

REFERENCE_PRESSURE = 20e-6      # Pa

EPSILON = 1e-12

# --------------------------------------------------------
# Basic Signal Statistics
# --------------------------------------------------------

def rms(signal: np.ndarray) -> float:
    """
    Root Mean Square.

    Parameters
    ----------
    signal : ndarray

    Returns
    -------
    float
    """

    return float(np.sqrt(np.mean(signal ** 2)))


def peak(signal: np.ndarray) -> float:
    """
    Peak amplitude.
    """

    return float(np.max(np.abs(signal)))


def crest_factor(signal: np.ndarray) -> float:
    """
    Crest Factor.

    Peak / RMS
    """

    return peak(signal) / (rms(signal) + EPSILON)


def energy(signal: np.ndarray) -> float:
    """
    Signal Energy.
    """

    return float(np.sum(signal ** 2))


# --------------------------------------------------------
# dB Conversions
# --------------------------------------------------------

def amplitude_to_db(amplitude):
    """
    Convert amplitude to dB.
    """

    amplitude = np.asarray(amplitude)

    return 20 * np.log10(np.maximum(amplitude, EPSILON))


def power_to_db(power):
    """
    Convert power to dB.
    """

    power = np.asarray(power)

    return 10 * np.log10(np.maximum(power, EPSILON))


def db_to_amplitude(db):
    """
    dB → amplitude.
    """

    return 10 ** (db / 20)


# --------------------------------------------------------
# Pressure
# --------------------------------------------------------

def pressure_to_spl(pressure):
    """
    Pascal → SPL.

    Parameters
    ----------
    pressure : float or ndarray

    Returns
    -------
    dB SPL
    """

    pressure = np.asarray(pressure)

    return 20 * np.log10(
        np.maximum(
            pressure,
            EPSILON
        ) / REFERENCE_PRESSURE
    )


def spl_to_pressure(spl):
    """
    SPL → Pascal.
    """

    return REFERENCE_PRESSURE * (10 ** (spl / 20))


# --------------------------------------------------------
# Normalization
# --------------------------------------------------------

def normalize(signal):
    """
    Normalize signal to ±1.
    """

    signal = np.asarray(signal)

    maximum = np.max(np.abs(signal))

    if maximum < EPSILON:

        return signal

    return signal / maximum


def remove_dc(signal):
    """
    Remove DC offset.
    """

    return signal - np.mean(signal)


# --------------------------------------------------------
# Window
# --------------------------------------------------------

def create_window(
    size: int,
    window="hann"
):
    """
    Create analysis window.
    """

    return get_window(window, size)


# --------------------------------------------------------
# Framing
# --------------------------------------------------------

def frame_signal(
    signal,
    frame_size,
    hop_size
):
    """
    Divide signal into overlapping frames.

    Returns
    -------
    ndarray

        shape

        (frames,
         frame_size)
    """

    signal = np.asarray(signal)

    number_frames = 1 + (

        len(signal) - frame_size

    ) // hop_size

    frames = np.zeros(

        (

            number_frames,

            frame_size

        )

    )

    for i in range(number_frames):

        start = i * hop_size

        stop = start + frame_size

        frames[i] = signal[start:stop]

    return frames


# --------------------------------------------------------
# Time Axis
# --------------------------------------------------------

def time_axis(
    signal,
    sample_rate
):
    """
    Time axis in seconds.
    """

    return np.arange(

        len(signal)

    ) / sample_rate


# --------------------------------------------------------
# FFT Frequency Axis
# --------------------------------------------------------

def frequency_axis(
    fft_size,
    sample_rate
):
    """
    Frequency axis.
    """

    return np.fft.rfftfreq(

        fft_size,

        d=1 / sample_rate

    )


# --------------------------------------------------------
# Next Power of Two
# --------------------------------------------------------

def next_power_of_two(value):
    """
    Returns next power of 2.

    Useful for FFT.
    """

    return 1 << (value - 1).bit_length()


# --------------------------------------------------------
# Signal Information
# --------------------------------------------------------

def signal_information(signal):
    """
    Quick signal statistics.

    Returns
    -------
    dict
    """

    return {

        "Length": len(signal),

        "Peak": peak(signal),

        "RMS": rms(signal),

        "Energy": energy(signal),

        "Crest Factor": crest_factor(signal)

    }


# --------------------------------------------------------
# Padding
# --------------------------------------------------------

def zero_pad(
    signal,
    length
):
    """
    Zero pad signal.
    """

    if len(signal) >= length:

        return signal

    output = np.zeros(length)

    output[:len(signal)] = signal

    return output


# --------------------------------------------------------
# Trim
# --------------------------------------------------------

def trim(
    signal,
    start,
    end
):
    """
    Trim samples.
    """

    return signal[start:end]


# --------------------------------------------------------
# Mono
# --------------------------------------------------------

def stereo_to_mono(signal):
    """
    Stereo → mono.

    Accepts shape

    (channels,samples)

    or

    (samples,channels)
    """

    signal = np.asarray(signal)

    if signal.ndim == 1:

        return signal

    if signal.shape[0] == 2:

        return np.mean(signal, axis=0)

    return np.mean(signal, axis=1)