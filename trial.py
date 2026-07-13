import os

os.environ["MPLCONFIGDIR"] = r"C:\SoundQualityFramework\temp"

os.makedirs(
    os.environ["MPLCONFIGDIR"],
    exist_ok=True
)

# ------------------------------------------------------
# Imports
# ------------------------------------------------------

import librosa

# from Core.Pre_Compliance_Check.standards import Standards
from Core.Pre_Compliance_Check.observations import AVASObservations


# ------------------------------------------------------
# Select Audio File
# ------------------------------------------------------

#audio_path = r"C:\SoundQualityFramework\data\avas_raw\NIS_BWD_MEAS.wav"

import librosa

from Core.Pre_Compliance_Check.fft_analysis import FFTAnalysis

audio = r"C:\SoundQualityFramework\data\avas_raw\NIS_BWD_MEAS.wav"

signal,sr = librosa.load(audio,sr=None,mono=True)

fft = FFTAnalysis(sr)

result = fft.analyze(signal)

fft.summary(result)

fft.plot(result)



observer = AVASObservations()

observer.print_summary(result)