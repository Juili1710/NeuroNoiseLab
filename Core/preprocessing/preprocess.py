# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:30:54 2026

@author: Lenovo
"""

"""
Signal Preprocessing Pipelines.
Ensures calculations do not alter original master tracks by returning modified ndarrays.
"""
import numpy as np
import librosa

def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Normalizes the audio signal to peak amplitude 1.0."""
    rms = np.max(np.abs(y))
    if rms > 0:
        return y / rms
    return y

def remove_dc_offset(y: np.ndarray) -> np.ndarray:
    """Removes DC bias component from the time series signal."""
    return y - np.mean(y)

def convert_to_mono(y: np.ndarray) -> np.ndarray:
    """Converts multi-channel array profiles into single-channel monaural tracks."""
    if y.ndim > 1:
        return librosa.to_mono(y)
    return y

def trim_silence(y: np.ndarray, top_db: int = 60) -> np.ndarray:
    """Trims leading and trailing silence regions based on decibel threshold."""
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt
