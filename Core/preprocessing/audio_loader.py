import librosa

def load_audio(path, target_sr=48000):

    audio, sr = librosa.load(
        path,
        sr=target_sr
    )

    return audio, sr


def get_audio_info(audio, sr):

    duration = len(audio) / sr

    return {
        "sample_rate": sr,
        "duration": duration,
        "samples": len(audio)
    }