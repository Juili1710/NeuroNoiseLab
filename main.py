import os

os.environ["MPLCONFIGDIR"] = r"C:\SoundQualityFramework\temp"

os.makedirs(
    os.environ["MPLCONFIGDIR"],
    exist_ok=True
)
import sys

from PySide6.QtWidgets import QApplication

from App.gui.main_window import MainWindow
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()