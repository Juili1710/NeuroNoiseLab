# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:19:52 2026

@author: Lenovo
"""

from preprocessing.batch_loader import get_audio_files
from preprocessing.audio_loader import load_audio
#from preprocessing.audio_loader import normalize
from features.extract_features import extract_basic_features

def build_dataset(folder):
#folder = "data/raw"

   files = get_audio_files(folder)
   dataset = []
   for file in files:
       signal, sr = load_audio(file)
       features = extract_basic_features(signal, sr)
       features["filepath"]=file
       dataset.append(features)
   return dataset
#print(dataset)
