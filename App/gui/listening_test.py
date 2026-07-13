"""
listening_test.py — Standalone AVAS Subjective Listening Test
Run: python listening_test.py
Plays each WAV file, shows a simple rating dialog, saves results to CSV.
Tested with PyQt5 + sounddevice.
"""

import sys
import csv
import random
import datetime
import soundfile as sf
import sounddevice as sd
import numpy as np
from pathlib import Path


from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGroupBox,
    QMessageBox, QProgressBar, QDialog,
    QLineEdit, QComboBox, QSpinBox,
    QDialogButtonBox, QFormLayout
)

from PySide6.QtCore import Qt


# ─────────────────────────────────────────────────
# CONFIGURE THESE BEFORE RUNNING
# ─────────────────────────────────────────────────
STIMULI_FOLDER = Path("C:/SoundQualityFramework/Data/avas_raw")        # folder containing your 10 WAV files
OUTPUT_CSV     = Path("C:/SoundQualityFramework/Data/metadata/jury_results.csv")
REPLAYS_ALLOWED = 5                      # max times listener can replay each clip


QUESTIONS = [
    # (question_text, scale_low_label, scale_high_label)
    ("Q1: How easily did you notice this as a vehicle warning signal?",
     "Not at all", "Immediately"),
    ("Q2: How strongly does this communicate urgency?",
     "Not urgent", "Very urgent"),
    ("Q3: How clearly could you tell approach vs moving-away direction?",
     "Not clear", "Very clear"),
    ("Q4: How annoying did you find this sound?",
     "Not annoying", "Very annoying"),
    ("Q5: How pleasant did you find this sound overall?",
     "Very unpleasant", "Very pleasant"),
    ("Q6: How natural or realistic did this sound seem?",
     "Artificial", "Very natural"),
    ("Q7: As a pedestrian, how acceptable is this as a warning signal?",
     "Unacceptable", "Fully acceptable"),
]

class ListenerInfoDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Participant Information")
        self.setMinimumWidth(350)

        layout = QFormLayout(self)
        self.all_results  = []
        self.responses = {}
        
        # Listener ID
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("e.g. L01")
        layout.addRow("Listener ID:", self.id_edit)

        # Age
        self.age_spin = QSpinBox()
        self.age_spin.setRange(10, 100)
        self.age_spin.setValue(25)
        layout.addRow("Age:", self.age_spin)

        # Gender
        self.gender_combo = QComboBox()
        self.gender_combo.addItems([
            "Female",
            "Male",
            "Non-binary",
            "Prefer not to say"
        ])
        layout.addRow("Gender:", self.gender_combo)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addRow(buttons)

    def get_data(self):
        return {
            "listener_id": self.id_edit.text().strip(),
            "age": self.age_spin.value(),
            "gender": self.gender_combo.currentText()
        }
class RatingWidget(QWidget):
    """Shows one stimulus at a time with replay button and 7 rating sliders."""

    def __init__(self, stimuli_paths):
        super().__init__()
        self.stimuli = stimuli_paths        # list of Path objects, already shuffled
        self.current_idx = 0
        self.replay_count = 0
        self.all_results  = []             # list of dicts, one per stimulus per subject
        self.responses = {}                # store responses for each stimulus index
        self.listener_id  = ""
        self.age = ""
        self.gender = ""
        self.audio_data   = None
        self.sample_rate  = None
        self.setWindowTitle("AVAS Listening Test — Sound Quality Study")
        self.setMinimumWidth(620)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Header ──────────────────────────────────────────────────────
        self.lbl_progress = QLabel()
        self.lbl_progress.setAlignment(Qt.AlignCenter)
        self.lbl_progress.setStyleSheet("font-size: 13px; color: gray;")
        layout.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Stimulus info + play button ──────────────────────────────────
        stim_group = QGroupBox("Current stimulus")
        stim_layout = QHBoxLayout(stim_group)

        self.lbl_stim_name = QLabel("—")
        self.lbl_stim_name.setStyleSheet("font-size: 14px; font-weight: bold;")
        stim_layout.addWidget(self.lbl_stim_name)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self.play_current)
        stim_layout.addWidget(self.btn_play)

        self.lbl_replays = QLabel()
        stim_layout.addWidget(self.lbl_replays)
        layout.addWidget(stim_group)

        # ── Rating sliders ───────────────────────────────────────────────
        ratings_group = QGroupBox("Rate this sound (1 = low, 5 = high)")
        ratings_layout = QVBoxLayout(ratings_group)

        self.sliders = []
        for q_text, low_label, high_label in QUESTIONS:
            row = QHBoxLayout()

            lbl = QLabel(q_text)
            lbl.setWordWrap(True)
            lbl.setFixedWidth(320)
            row.addWidget(lbl)

            row.addWidget(QLabel(low_label))

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(1)
            slider.setMaximum(5)
            slider.setValue(3)              # start at midpoint
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(1)
            slider.setFixedWidth(140)
            row.addWidget(slider)

            row.addWidget(QLabel(high_label))

            val_lbl = QLabel("3")           # shows current slider value
            val_lbl.setFixedWidth(16)
            val_lbl.setAlignment(Qt.AlignCenter)
            slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            row.addWidget(val_lbl)

            ratings_layout.addLayout(row)
            self.sliders.append(slider)

        layout.addWidget(ratings_group)

        # ── Navigation ───────────────────────────────────────────────────
        nav_layout = QHBoxLayout()
        self.btn_back = QPushButton(
            "← Previous"
        )

        nav_layout.addWidget(
            self.btn_back
        )

        self.btn_back.clicked.connect(
            self.previous_stimulus
        )
        self.btn_next = QPushButton("Next stimulus →")
        self.btn_next.setFixedHeight(36)
        self.btn_next.clicked.connect(self.next_stimulus)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        # self.lbl_save_status = QLabel(
        #     "Not Saved"
        # )
        # layout.addWidget(
        #     self.lbl_save_status
        # )
        
        self.update_display()

    def update_display(self):
        n     = len(self.stimuli)
        idx   = self.current_idx
        total = n

        self.lbl_progress.setText(f"Stimulus {idx + 1} of {total}")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(idx)

        stim_path = self.stimuli[idx]
        self.lbl_stim_name.setText(stim_path.stem)   # filename without extension

        self.replay_count = 0
        self.lbl_replays.setText(f"Plays remaining: {REPLAYS_ALLOWED}")
        self.btn_play.setEnabled(True)

        # Load audio
        try:
            data, sr = sf.read(str(stim_path), always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            self.audio_data  = data.astype(np.float64)
            self.sample_rate = sr
        except Exception as e:
            QMessageBox.critical(self, "Load Error",
                                 f"Could not load {stim_path.name}:\n{e}")

        # Restore previous ratings if they exist,
        # otherwise initialise sliders to midpoint.
        if self.current_idx in self.responses:

            row = self.responses[self.current_idx]
            print("--------------------------------")
            print("Current:", self.current_idx)
            print("Saved keys:", self.responses.keys())
            for i, slider in enumerate(self.sliders):
                slider.setValue(row[f"Q{i+1}"])

        else:

            for slider in self.sliders:
                slider.setValue(3)

        # Auto-play first time
        self.play_current()

    def play_current(self):
        if self.replay_count >= REPLAYS_ALLOWED:
            self.btn_play.setEnabled(False)
            return
        try:
            sd.stop()
            sd.play(self.audio_data, self.sample_rate)
            self.replay_count += 1
            remaining = REPLAYS_ALLOWED - self.replay_count
            self.lbl_replays.setText(f"Plays remaining: {remaining}")
            if remaining == 0:
                self.btn_play.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", str(e))

    def next_stimulus(self):
        if self.current_idx >= len(self.stimuli):
            return

        sd.stop()
        # Save current ratings
        stim_path = self.stimuli[self.current_idx]
        row = {
            "listener_id":  self.listener_id,
            "age": self.age,
            "gender": self.gender,
            "stimulus":     stim_path.stem,
            "timestamp":    datetime.datetime.now().isoformat(),
        }
        for i, (q_text, _, _) in enumerate(QUESTIONS):
            q_key = f"Q{i+1}"
            row[q_key] = self.sliders[i].value()

        #self.all_results.append(row)
        if self.current_idx < len(self.all_results):

            self.all_results[self.current_idx] = row

        else:

            self.all_results.append(row)
        # Save ratings in memory for Back button
        self.responses[self.current_idx] = row
        
        self.current_idx += 1
        if self.current_idx >= len(self.stimuli):
            self.save_and_finish()
            self.btn_next.setEnabled(False)
        else:
            self.update_display()
        print("Current index:", self.current_idx)
        print("Length all_results:", len(self.all_results))
        print("Length responses:", len(self.responses))

    def save_and_finish(self):
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        # Append to CSV (creates header if file doesn't exist)
        file_exists = OUTPUT_CSV.exists()
        with open(OUTPUT_CSV, "a", newline="") as f:
            fieldnames = ["listener_id", "age","gender","stimulus", "timestamp"] + \
                         [f"Q{i+1}" for i in range(len(QUESTIONS))]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()       # write header only once
            # Write the latest response for every stimulus
            for idx in range(len(self.stimuli)):
                if idx in self.responses:
                    writer.writerow(self.responses[idx])

        QMessageBox.information(
            self, "Test Complete",
            f"Thank you! Results saved to:\n{OUTPUT_CSV}"
        )
        self.close()
    def previous_stimulus(self):

        if self.current_idx == 0:
            return

        # -------- Save current ratings before leaving --------
        stim_path = self.stimuli[self.current_idx]

        row = {
            "listener_id": self.listener_id,
            "age": self.age,
            "gender": self.gender,
            "stimulus": stim_path.stem,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        for i, (q_text, _, _) in enumerate(QUESTIONS):
            row[f"Q{i+1}"] = self.sliders[i].value()

        self.responses[self.current_idx] = row

        if self.current_idx < len(self.all_results):
            self.all_results[self.current_idx] = row
        else:
            self.all_results.append(row)

        # -------- Now go back --------
        sd.stop()

        self.current_idx -= 1

        self.update_display()


def launch_jury_test(parent=None):
     #Show instructions
    QMessageBox.information(
        None, "Instructions",
        "You will hear 10 AVAS (Acoustic Vehicle Alerting System) sounds.\n\n"
        "For each sound:\n"
        "1. Press ▶ Play to listen (up to 5 times)\n"
        "2. Rate the sound on all 7 scales (ask if you dont understand any scale question)\n"
        "3. Press 'Next stimulus →' to continue\n\n"
        "Sit quietly. Take your time.\n\n"
        "Press OK to begin."
    )
    dialog = ListenerInfoDialog()

    if dialog.exec() != QDialog.Accepted:
        return None

    listener_data = dialog.get_data()

    wav_files = sorted(
        STIMULI_FOLDER.glob("*.wav")
    )

    if not wav_files:

        QMessageBox.critical(
            parent,
            "No Files Found",
            f"No WAV files found in\n{STIMULI_FOLDER}"
        )

        return None

    random.shuffle(wav_files)

    win = RatingWidget(wav_files)

    win.listener_id = listener_data["listener_id"]
    win.age = listener_data["age"]
    win.gender = listener_data["gender"]

    win.show()

    return win


