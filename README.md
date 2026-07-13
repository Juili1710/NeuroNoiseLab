# NeuroNoiseLab
An open-source scalable Python framework for sound quality modelling using psychoacoustic analysis, jury listening tests and machine learning.
# Sound Quality Framework

## Overview

Sound Quality Framework is an open-source Python framework for objective and subjective sound quality analysis.

The framework integrates

- Audio preprocessing
- Psychoacoustic feature extraction
- Jury listening tests
- Preliminary AVAS compliance checking
- Machine learning based sound quality prediction

The framework is designed to be scalable for future sound design applications including

- Electric Vehicle AVAS
- Electric Motor Noise
- Sonification
- Hospital alarms
- and many more

---

## Features

✓ Batch audio loading

✓ Audio preprocessing

✓ Time/Frequency visualization

✓ Spectrogram generation

✓ Psychoacoustic metrics

- Loudness
- Sharpness
- Roughness
- (Extensible)

✓ Jury listening test GUI

✓ Jury analysis

✓ AVAS analysis

✓ Training dataset generation

✓ Machine learning

- Multiple Linear Regression
- Artificial Neural Network

✓ Model comparison

---

## GUI

---
## Audio Visualization

Displays waveform, spectrogram and audio information.

![Visualization](Docs/screenshots/Visulization%20Tab.png)
## Psychoacoustic Analysis

Extract objective sound quality metrics such as Loudness, Sharpness and Roughness.

![Features](Docs/screenshots/Psychoaccoustics%20features.png)
## Jury Listening Test

Interactive listening test interface for collecting subjective ratings.

![Listening Test](Docs/screenshots/Jury%20test.png)
## AI / Machine Learning

Generate datasets, train prediction models and compare their performance.

![AI Module](Docs/screenshots/ML%20model%20tab.png)

## 📁 Project Structure

```text
SoundQualityFramework
│
├── App
│   └── gui
│       ├── main_window.py
│       └── listening_test.py
│
├── Core
│   ├── preprocessing
│   ├── visualizations
│   ├── features
│   │   ├── psychoacoustics.py
│   │   └── export_features.py
│   │
│   ├── jury_tests_analysis
│   │
│   ├── models
│   │   ├── create_dataset.py
│   │   ├── evaluator.py
│   │   ├── validation.py
│   │   ├── MLR.py
│   │   ├── ann.py
│   │   └── compare_models.py
│   │
│   └── Pre_Compliance_Check
│       ├── recording_checks.py
│       ├── octave_analysis.py
│       ├── standards.py
│       └── compliance_engine.py
│
├── Data
│   ├── raw
│   ├── processed
│   ├── avas_raw
│   └── metadata
│
├── Docs
│   ├── screenshots
│   └── R138r1am4e.pdf
│
├── Results
│
├── temp
│
├── main.py
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/USERNAME/SoundQualityFramework.git

cd SoundQualityFramework

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Future Work

- Random Forest
- Support Vector Regression
- XGBoost
- Hyperparameter Optimization
- Web Application
- Explainable AI
- Automatic Sound Design Guidance

---

## Citation

If you use this framework, please cite the associated publication.

---

## License

MIT License
