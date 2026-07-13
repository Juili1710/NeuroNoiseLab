# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:29:59 2026

@author: Lenovo
"""

from pathlib import Path
audio_extensions = [
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a"
    ]
def get_audio_files(folder_path):

   
    files = []

    for ext in audio_extensions:

        files.extend(
            Path(folder_path).rglob(f"*{ext}")
        )

    return sorted(files)
