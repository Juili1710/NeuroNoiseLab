from unittest import result

import soundfile as sf
import librosa
import numpy as np
from mosqito.sq_metrics import (loudness_zwtv)
from mosqito.sq_metrics import (sharpness_din_from_loudness)
from mosqito.sq_metrics import roughness_dw
#from mosqito.sq_metrics import tonality_tnr_pr
#from mosqito.sq_metrics import fs
def extract_all_features(filepath):

    signal, fs = sf.read(filepath)

    features = {}
    ##resampling
    if signal.ndim > 1:
            signal = signal.mean(axis=1)
    if fs != 48000:
        signal = librosa.resample(
                signal,
                orig_sr=fs,
                target_sr=48000
            )
        fs = 48000
    try:
        
            ## loudness calculation
        N, N_spec, bark_axis, time_axis = loudness_zwtv(signal,fs)

        features["Loudness"] = round(
            float(N.mean()),
            2
        )
        features["N"] = N
        features["time_axis"] = time_axis
        features["N_spec"] = N_spec
        features["bark_axis"] = bark_axis

    except Exception as e:

        features["Loudness"] = f"Error: {e}"
    ## sharpness calculation
    try:

        S = sharpness_din_from_loudness(N, N_spec)

        features["Sharpness"] = round(float(np.mean(S)),2)

    except Exception as e:

        features["Sharpness"] = f"Error: {e}"
    ## roughness calculation
    try:

        R = roughness_dw(signal, fs)
        if isinstance(R, tuple):
            R = R[0]
        features["Roughness"] = round(float(np.mean(R)),2)

    except Exception as e:

        features["Roughness"] = f"Error: {e}"
    ## tonality calculation
    # try:    
    #     tnr, pr = tonality_tnr_pr(signal, fs=fs)
    #     features["TNR"] = round(float(tnr),2)
    # except Exception as e:
    #     features["TNR"] = f"Error: {e}"

    #Fuctuation strenght calculation 
    # try:
    #     fluct = fs(audio, fs=sr)
    # except Exception as e:

    #     features["Fluctuation Strength"] = f"Error: {e}"
    return features