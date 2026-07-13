# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:33:51 2026

@author: Lenovo
"""

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from App.gui.listening_test import launch_jury_test
from PySide6.QtWidgets import QFileDialog
import librosa
import numpy as np
import pyqtgraph as pg
import librosa.display
import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(),"temp")
os.makedirs(os.environ['MPLCONFIGDIR'],exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Core.jury_tests_analysis.jury_analysis import JuryAnalysis

from Core.models.MLR import train_multiple_linear_regression
from Core.models.ann import train_ann
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from Core.Pre_Compliance_Check.fft_analysis import FFTAnalysis
from Core.Pre_Compliance_Check.observations import AVASObservations 
from Core.features.batch_feature_extraction import batch_extract_features
from Core.datasets.training_dataset_builder import (build_training_dataset)



class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()
        self.setWindowTitle("Open Sound Quality Framework")
        self.resize(1400, 800)
        self.setup_ui()
        self.jury_analysis = JuryAnalysis()
        

    def setup_ui(self):

        # ==========================
        # Main Widget & Layout
        # ==========================
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # ==========================
        # LEFT PANEL
        # ==========================
        left_panel = QVBoxLayout()

        # --------------------------
        # Dataset Group
        # --------------------------
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout()

        self.load_file_btn = QPushButton("Load Audio File")

        self.jury_test_btn = QPushButton("Conduct Jury Test")

        
        self.jury_analysis_btn = QPushButton("Load Jury CSV")
        dataset_layout.addWidget(self.load_file_btn)

        dataset_layout.addWidget(self.jury_test_btn)
        dataset_group.setLayout(dataset_layout)

        # --------------------------
        # Features Group
        # --------------------------
        feature_group = QGroupBox("Psychoacoustic Features")

        feature_layout = QVBoxLayout()

        self.feature_list = QListWidget()

        self.feature_list.addItems([
            "Loudness",
            "Sharpness",
            "Roughness"
            ])

        feature_layout.addWidget(self.feature_list)

        feature_group.setLayout(feature_layout)

        # --------------------------
        # Extract Button
        # --------------------------
        self.extract_btn = QPushButton(
            "Extract Features"
        )

        # --------------------------
        # Add Everything to Left Panel
        # --------------------------
        left_panel.addWidget(dataset_group)

        left_panel.addWidget(feature_group)

        left_panel.addWidget(self.extract_btn)
        left_panel.addWidget(self.jury_analysis_btn)

        left_panel.addStretch()

        # ==========================
        # RIGHT PANEL
        # ==========================
        self.tabs = QTabWidget()

        self.visualization_tab = QWidget()
        
        ## Preliminary Compliance Tab
        self.avas_tab = QWidget()
        

        
        ## Features Tab
        self.features_tab = QWidget()
        features_layout = QVBoxLayout()
        self.features_text = QTextEdit()
        self.features_text.setReadOnly(True)
        self.loudness_plot = pg.PlotWidget()
        features_layout.addWidget(self.features_text)
        features_layout.addWidget(self.loudness_plot)
        self.features_tab.setLayout(features_layout)
        
        ## Jury Statistics Tab
        self.statistics_tab = QWidget()
        stats_layout = QVBoxLayout()

        self.stats_text = QLabel("No Jury Data Loaded")

        stats_layout.addWidget(
            self.stats_text
        )
        ####################################################
        # AI / ML TAB
        ####################################################

        self.ai_tab = QWidget()

        

        stats_layout = QVBoxLayout()

        self.stats_text = QLabel("No Jury Data Loaded")

        stats_layout.addWidget(
            self.stats_text
        )
        


        self.statistics_tab.setLayout(stats_layout)
        
        self.demo_btn= QPushButton("Demographics")
        
        self.StimAna_btn = QPushButton("Stimulus Analysis") 
        
        self.QAna_btn = QPushButton("Questionnaire Analysis")

        self.tabs.addTab(
            self.visualization_tab,
            "Visualization"
        )
        self.tabs.addTab(
            self.avas_tab,
            "AVAS Evaluation"
        )
                # ==========================
        # Visualization Tab Layout
        # ==========================

        vis_layout = QVBoxLayout()

        self.waveform_plot = pg.PlotWidget()
        self.spec_figure = Figure()
        self.spec_canvas = FigureCanvasQTAgg(self.spec_figure)
        vis_layout.addWidget(self.spec_canvas)
        vis_layout.addWidget(self.waveform_plot)

        self.visualization_tab.setLayout(vis_layout)
        # ==========================
        avas_layout = QVBoxLayout()

        self.avas_tab.setLayout(avas_layout)

        ########################################################

        self.run_fft_btn = QPushButton(

            "Run AVAS Evaluation"

        )

        avas_layout.addWidget(

            self.run_fft_btn

        )

        ########################################################

        self.fft_plot = pg.PlotWidget()

        avas_layout.addWidget(

            self.fft_plot

        )

        ########################################################

        self.fft_summary = QTextEdit()

        self.fft_summary.setReadOnly(True)

        avas_layout.addWidget(

            self.fft_summary

        )

        ########################################################

        self.run_fft_btn.clicked.connect(

            self.run_avas_evaluation

        )
        # ==========================
        # Features-obj psychoaccoustics Tab Layout
        # ==========================
        self.tabs.addTab(
            self.features_tab,
            "Features"
        )
        # ==========================
        # Jury Statistics Tab Layout
        # ==========================
        self.tabs.addTab(
            self.statistics_tab,
            "Jury Statistics"
        )
        
        stats_layout.addWidget(self.demo_btn)
        stats_layout.addWidget(self.StimAna_btn)
        stats_layout.addWidget(self.QAna_btn)
        self.compliance_tab = QWidget()
        ############################################
        # AI TAB LAYOUT
        ####################################################
        self.tabs.addTab(self.ai_tab,"AI / ML")
        ai_layout = QVBoxLayout()

        self.ai_tab.setLayout(ai_layout)
        target_layout = QHBoxLayout()

        target_layout.addWidget(QLabel("Target Variable"))

        self.target_combo = QComboBox()

        self.target_combo.addItems([
            "-- Select Target Variable --",
            "Noticeability",
            "Urgency",
            "Detectability",
            "Annoyance",
            "Pleasantness",
            "Naturalness",
            "Acceptability"

        ])

        target_layout.addWidget(self.target_combo)

        ai_layout.addLayout(target_layout)
        

        ####################################################
        # Prediction Models
        ####################################################

        ai_layout.addWidget(QLabel("Prediction Models"))

        models_layout = QHBoxLayout()

        self.mlr_cb = QCheckBox("MLR")
        self.ann_cb = QCheckBox("ANN")
        self.rf_cb = QCheckBox("RF")
        self.svr_cb = QCheckBox("SVR")
        self.xgb_cb = QCheckBox("XGBoost")

        self.mlr_cb.setChecked(True)

        models_layout.addWidget(self.mlr_cb)
        models_layout.addWidget(self.ann_cb)
        models_layout.addWidget(self.rf_cb)
        models_layout.addWidget(self.svr_cb)
        models_layout.addWidget(self.xgb_cb)
        models_layout.addStretch()

        ai_layout.addLayout(models_layout)
        validation_layout = QHBoxLayout()

        validation_layout.addWidget(QLabel("Validation"))

        self.validation_combo = QComboBox()

        self.validation_combo.addItems([

            "LOOCV",

            "Train/Test Split",

            "K-Fold Cross Validation",

            "Stratified K-Fold"

        ])

        validation_layout.addWidget(self.validation_combo)

        ai_layout.addLayout(validation_layout)
        button_layout = QHBoxLayout()

        self.prepare_dataset_btn = QPushButton("Prepare Dataset")
        self.prepare_dataset_btn.clicked.connect(self.prepare_dataset)

        self.train_model_btn = QPushButton("Train Model")

        self.compare_models_btn = QPushButton("Compare Models")
        ##buttons 
        button_layout.addWidget(self.prepare_dataset_btn)

        button_layout.addWidget(self.train_model_btn)

        button_layout.addWidget(self.compare_models_btn)

        ai_layout.addLayout(button_layout)
        
        #comparison Table
        ai_layout.addWidget(
            QLabel("Model Comparison")
        )
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(6)
        self.model_table.setMaximumHeight(120)
        self.model_table.setHorizontalHeaderLabels([

            "Rank",

            "Model",

            "Validation",

            "R²",

            "RMSE",

            "MAE"

        ])

        ai_layout.addWidget(
            self.model_table
        )
        ##Model Summary
        ai_layout.addWidget(QLabel("Model Summary"))

        self.model_summary = QTextEdit()

        self.model_summary.setReadOnly(True)

        self.model_summary.setMinimumHeight(300)

        ai_layout.addWidget(self.model_summary)
        self.model_summary.setPlainText(
        """No model trained.

        Workflow:

        1. Prepare Dataset
        2. Select Target Variable
        3. Select ONE Prediction Model
        4. Train Model
        5. Compare Models
        """
        )
        

        
            

        # ==========================
        # Final Layout
        # ==========================
        main_layout.addLayout(left_panel, 1)

        main_layout.addWidget(self.tabs, 5)

        # ==========================
        # SIGNAL CONNECTIONS
        # ==========================

        # Load Audio Button
        self.load_file_btn.clicked.connect(self.load_audio_file)

        # Conduct Jury Test Button
        self.jury_test_btn.clicked.connect(self.open_jury_test)

        # Extract Features Button
        self.extract_btn.clicked.connect(self.extract_features)
        #jury_analysis button 
        self.jury_analysis_btn.clicked.connect(self.load_jury_csv)
        
        self.demo_btn.clicked.connect(
            lambda:(
            self.jury_analysis.gender_pie(),
            self.jury_analysis.age_histogram()
            )
        )
        self.StimAna_btn.clicked.connect(
            lambda:(
            self.jury_analysis.stimulus_comparison(), # annoyance vs Pleasantness
            self.jury_analysis.detectability_per_stimulus(),
            self.jury_analysis.pleasantness_per_stimulus(),
            self.jury_analysis.annoyance_per_stimulus()
            )
        )
        self.QAna_btn.clicked.connect(
            lambda:(
            self.jury_analysis.mean_ratings(),
            #self.jury_analysis.annoyance_distribution()
            )
          
        )
        
        #Ai-ml buttons
        self.train_model_btn.clicked.connect(self.train_selected_model)
        self.compare_models_btn.clicked.connect(self.compare_selected_models)
        
        self.status = QStatusBar()

        self.setStatusBar(self.status)

        self.status.showMessage("Ready")
    def open_jury_test(self):
        self.jury_window = launch_jury_test(self)
    def load_audio_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac)"
        )

        if not filename:
            return

        try:

            self.signal, self.sr = librosa.load(
                filename,
                sr=None,
                mono=True
            )

            self.audio_path = filename
            self.plot_waveform()
            self.plot_spectrogram()
            QMessageBox.information(
                self,
                "Audio Loaded",
                f"File:\n{filename}\n\n"
                f"Sampling Rate: {self.sr} Hz\n"
                f"Samples: {len(self.signal)}"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )
    
    def plot_waveform(self):
        if not hasattr(self, "signal"):
            return

        self.waveform_plot.clear()

        x = np.arange(
            len(self.signal)
        ) / self.sr

        self.waveform_plot.plot(
            x,
            self.signal
        )

        self.waveform_plot.setLabel(
            "bottom",
            "Time (s)"
        )

        self.waveform_plot.setLabel(
            "left",
            "Amplitude"
        )

        self.waveform_plot.setTitle(
            "Audio Waveform"
        )  
        self.waveform_plot.getViewBox().setMouseEnabled(
            x=True,
            y=False
        )
        self.waveform_plot.setLimits(
            yMin=-1.1,
            yMax=1.1
        )
    def plot_spectrogram(self):

        
        if not hasattr(self, "signal"):
            return

        self.spec_figure.clear()

        ax = self.spec_figure.add_subplot(111)

        D = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(self.signal)
            ),
            ref=np.max
        )

        img = librosa.display.specshow(
            D,
            sr=self.sr,
            x_axis="time",
            y_axis="log",
            cmap="viridis",
            ax=ax
        )

        ax.set_title(
            "Spectrogram"
        )

        self.spec_figure.colorbar(
            img,
            ax=ax,
            format="%+2.0f dB"
        )

        self.spec_canvas.draw()
    def extract_features(self):
        
        if not hasattr(
            self,
            "audio_path"
        ):

            QMessageBox.warning(
                self,
                "No Audio",
                "Please load an audio file first."
            )

            return

        from Core.features.psychoacoustics import (
            extract_all_features
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        features = extract_all_features(
            self.audio_path
        )
        QApplication.restoreOverrideCursor()
        #print(features)
        ### Plot specific loudness with loudness vs bark 
        if ("N_spec" in features and "bark_axis" in features):

            self.plot_specific_loudness(
                features["N_spec"],
                features["bark_axis"]
            )
        text = ""

        for key, value in features.items():

            # Don't print arrays
            if key in [
                "N",
                "N_spec",
                "time_axis",
                "bark_axis"
            ]:
                continue

            if key == "Loudness":

                if isinstance(value, (int, float)):
                    text += f"{key}: {value:.2f} sones\n"
                else:
                    text += f"{key}: {value}\n"

            elif key == "Sharpness":

                if isinstance(value, (int, float)):
                    text += f"{key}: {value:.2f} acum\n"
                else:
                    text += f"{key}: {value}\n"

            else:
                text += f"{key}: {value}\n"

        self.features_text.setText(text)
    def plot_loudness(self, N, time_axis):


        self.loudness_plot.clear()

        self.loudness_plot.plot(
            time_axis,
            N
        )

        self.loudness_plot.setTitle(
            "Time-Varying Loudness"
        )

        self.loudness_plot.setLabel(
            "left",
            "Loudness (sones)"
        )

        self.loudness_plot.setLabel(
            "bottom",
            "Time (s)"
        )
    def plot_specific_loudness(self, N_spec, bark_axis):


        self.loudness_plot.clear()

        mean_spec = np.mean(
            N_spec,
            axis=1
        )

        self.loudness_plot.plot(
            bark_axis,
            mean_spec
        )

        self.loudness_plot.setTitle(
            "Specific Loudness"
        )

        self.loudness_plot.setLabel(
            "bottom",
            "Critical Band (Bark)"
        )

        self.loudness_plot.setLabel(
            "left",
            "Specific Loudness (sone/Bark)"
        )
    def load_jury_csv(self):

        file_path, _ = QFileDialog.getOpenFileName(
        self,
        "Select Jury Results",
        "",
        "CSV Files (*.csv)")

        if file_path:

            self.jury_analysis.load_csv(
                file_path
            )

            QMessageBox.information(
                self,
                "Success",
                f"Loaded:\n{file_path}"
            )
    
    def get_selected_models(self):


        models = []

        if self.mlr_cb.isChecked():
            models.append("MLR")

        if self.ann_cb.isChecked():
            models.append("ANN")

        if self.rf_cb.isChecked():
            models.append("RF")

        if self.svr_cb.isChecked():
            models.append("SVR")

        if self.xgb_cb.isChecked():
            models.append("XGB")

        return models
    def train_selected_model(self):     
        selected = self.get_selected_models()
        if self.target_combo.currentIndex() == 0:
            QMessageBox.warning(
                self,
                "Target Variable",
                "Please select a target variable."
            )
            return
        if len(selected) != 1:
            QMessageBox.warning(
                self,
                "Training",
                "Please select exactly ONE model."
            )
            return
        model = selected[0]
        csv_file = getattr(

            self,

            "training_dataset",

            "Data/metadata/training_dataset.csv"

        )
        input_features = [
            "Loudness",
            "Sharpness",
            "Roughness"
        ]
        target = self.target_combo.currentText()
        validation = self.validation_combo.currentText()
        if validation != "LOOCV":
            QMessageBox.information(
                self,
                "Validation",
                "Only LOOCV implemented currently."
            )

            return

        if model == "MLR":
            results = train_multiple_linear_regression(
                csv_file=csv_file,
                input_features=input_features,
                target=target,
                validation="LOOCV"
            )

            self.display_mlr_summary(results)
            self.update_prediction_plot(

                results["Results"]["Actual"],

                results["Results"]["Predicted"],

                "Multiple Linear Regression"

            )

        elif model == "ANN":
            results = train_ann(
                csv_file=csv_file,
                input_features=input_features,
                target=target,
                validation="LOOCV"
            )
            self.display_ann_summary(results)
            self.update_prediction_plot(

                results["Results"]["Actual"],

                results["Results"]["Predicted"],

                "Artificial Neural Network"

            )
        else:
            QMessageBox.information(
                self,
                "Coming Soon",
                f"{model} not implemented yet."
            )
    
       
    def display_mlr_summary(self, results):

        summary = f"""

    Model
    ----------------------------
    Multiple Linear Regression

    Target
    ----------------------------
    {self.target_combo.currentText()}

    Validation
    ----------------------------
    LOOCV

    Equation
    ----------------------------
    {results["Equation"]}

    """

        self.model_summary.setPlainText(summary)
    def display_ann_summary(self, results):

        summary = f"""

    Model
    ----------------------------
    Artificial Neural Network

    Target
    ----------------------------
    {self.target_combo.currentText()}

    Validation
    ----------------------------
    LOOCV

    Architecture
    ----------------------------
    3 - 5 - 1

    Activation
    ----------------------------
    ReLU

    Optimizer
    ----------------------------
    Adam

    """

        self.model_summary.setPlainText(summary)
    def update_prediction_plot(self, actual, predicted, model_name):
        """
        Display prediction plot in a separate window.
        """

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))

        plt.scatter(
            actual,
            predicted,
            s=70
        )

        mn = min(min(actual), min(predicted))
        mx = max(max(actual), max(predicted))

        plt.plot(
            [mn, mx],
            [mn, mx],
            "r--",
            linewidth=2
        )

        plt.xlabel("Actual Rating")
        plt.ylabel("Predicted Rating")

        plt.title(
            f"{model_name}\nActual vs Predicted"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.show(block=False)

    def run_avas_evaluation(self):


        if not hasattr(self, "signal"):

            QMessageBox.warning(

                self,

                "Audio",

                "Load an audio file first."

            )

            return

        ##################################################

        fft = FFTAnalysis(

            self.sr

        )

        result = fft.analyze(

            self.signal

        )

        ##################################################

        self.fft_plot.clear()

        self.fft_plot.plot(

            result.frequencies,

            result.magnitude

        )

        self.fft_plot.setLabel(

            "bottom",

            "Frequency (Hz)"

        )

        self.fft_plot.setLabel(

            "left",

            "Magnitude"

        )

        self.fft_plot.setTitle(

            "FFT Spectrum"

        )

        ##################################################

        observer = AVASObservations()

        observations = observer.generate(

            result

        )

        text = ""

        for item in observations:

            text += "• " + item + "\n"

        self.fft_summary.setPlainText(

            text

        )

   
    
    def append_prepare_log(self, text):

        self.model_summary.append(text)
        # Force GUI update
        
        QApplication.processEvents()
    def prepare_dataset(self):

        QMessageBox.information(

            self,

            "Prepare Dataset",

            "Select the folder containing all AVAS WAV files."

        )

        folder = QFileDialog.getExistingDirectory(

            self,

            "Select Dataset Folder"

        )

        if not folder:

            return

        self.model_summary.clear()

        self.model_summary.append(

            "Preparing Training Dataset...\n"

        )

        QApplication.processEvents()

        # ----------------------------------------

        objective_csv = "Data/metadata/objective_features.csv"

        jury_csv = "Data/metadata/jury_results.csv"

        training_csv = "Data/metadata/training_dataset.csv"

        # ----------------------------------------

        self.model_summary.append(

            "Step 1/3 : Extracting objective features..."

        )

        QApplication.processEvents()

        batch_extract_features(

            folder,

            objective_csv,

            progress_callback=self.append_prepare_log

        )

        # ----------------------------------------

        self.model_summary.append(

            "\nStep 2/3 : Merging with jury ratings..."

        )

        QApplication.processEvents()

        build_training_dataset(

            objective_csv,

            jury_csv,

            training_csv

        )

        # ----------------------------------------

        self.training_dataset = training_csv

        self.model_summary.append(

            "\n✓ Training dataset created."

        )

        self.model_summary.append(

            f"\nSaved to:\n{training_csv}"

        )

        QApplication.processEvents()

        QMessageBox.information(

            self,

            "Done",

            "Training dataset prepared successfully."

        )
    def compare_selected_models(self):

        selected = self.get_selected_models()
        if self.target_combo.currentIndex() == 0:
            QMessageBox.warning(
                self,
                "Target Variable",
                "Please select a target variable."
            )
            return
        if len(selected) < 2:

            QMessageBox.warning(
                self,
                "Compare Models",
                "Select at least TWO models."
            )
            return

        csv_file = getattr(
            self,
            "training_dataset",
            "Data/metadata/training_dataset.csv"
        )

        input_features = [
            "Loudness",
            "Sharpness",
            "Roughness"
        ]

        target = self.target_combo.currentText()

        validation = self.validation_combo.currentText()

        results = []

        # ------------------------------
        # Train selected models
        # ------------------------------

        if "MLR" in selected:

            r = train_multiple_linear_regression(
                csv_file,
                input_features,
                target,
                validation
            )

            r["Name"] = "Multiple Linear Regression"

            results.append(r)

        if "ANN" in selected:

            r = train_ann(
                csv_file,
                input_features,
                target,
                validation
            )

            r["Name"] = "Artificial Neural Network"
            print(r)
            results.append(r)

        # Future
        # RF
        # SVR
        # XGBoost

        # ------------------------------
        # Sort by R²
        # ------------------------------

        results.sort(

            key=lambda x: x["Metrics"]["R2"],

            reverse=True

        )

        # ------------------------------
        # Populate table
        # ------------------------------

        self.model_table.setRowCount(len(results))

        medals = ["🥇", "🥈", "🥉"]

        comparison = []

        for row, r in enumerate(results):

            m = r["Metrics"]

            rank = medals[row] if row < 3 else str(row + 1)

            self.model_table.setItem(
                row,
                0,
                QTableWidgetItem(rank)
            )

            self.model_table.setItem(
                row,
                1,
                QTableWidgetItem(r["Name"])
            )

            self.model_table.setItem(
                row,
                2,
                QTableWidgetItem(validation)
            )

            self.model_table.setItem(
                row,
                3,
                QTableWidgetItem(f"{m['R2']:.3f}")
            )

            self.model_table.setItem(
                row,
                4,
                QTableWidgetItem(f"{m['RMSE']:.3f}")
            )

            self.model_table.setItem(
                row,
                5,
                QTableWidgetItem(f"{m['MAE']:.3f}")
            )

            comparison.append({

                "Rank": row + 1,

                "Model": r["Name"],

                "Validation": validation,

                "R2": m["R2"],

                "RMSE": m["RMSE"],

                "MAE": m["MAE"]

            })

        # ------------------------------
        # Save comparison CSV
        # ------------------------------

        import pandas as pd

        pd.DataFrame(comparison).to_csv(

            "Data/metadata/model_comparison.csv",

            index=False

        )

        # ------------------------------
        # Summary
        # ------------------------------

        best = results[0]

        self.model_summary.append("\n")

        self.model_summary.append("===== Model Comparison =====")

        self.model_summary.append(

            f"Best Model : {best['Name']}"

        )

        self.model_summary.append(

            f"Target : {target}"

        )

        self.model_summary.append(

            f"Validation : {validation}"

        )

        self.model_summary.append(

            f"R² = {best['Metrics']['R2']:.3f}"

        )

        self.model_summary.append(

            f"RMSE = {best['Metrics']['RMSE']:.3f}"

        )

        self.model_summary.append(

            f"MAE = {best['Metrics']['MAE']:.3f}"

        )

        self.model_summary.append(

            "\nComparison saved to\nData/metadata/model_comparison.csv"

        )

        QMessageBox.information(

            self,

            "Comparison Complete",

            f"Best model: {best['Name']}"

        )