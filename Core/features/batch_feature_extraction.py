"""
Batch Psychoacoustic Feature Extraction

Generates one CSV for ML training.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from Core.features.psychoacoustics import extract_all_features


def batch_extract_features(
        dataset_folder,
        output_csv,
        progress_callback=None
):
    """
    Batch extract psychoacoustic features from WAV files.

    Parameters
    ----------
    dataset_folder : str
        Folder containing WAV files.

    output_csv : str
        Output CSV path.

    progress_callback : callable, optional
        Function used to display progress in GUI.
        Example:
            progress_callback("Processing file.wav")
    """

    def log(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)

    dataset_folder = Path(dataset_folder)

    rows = []

    wav_files = sorted(dataset_folder.glob("*.wav"))

    log("=" * 60)
    log("Batch Feature Extraction")
    log("=" * 60)
    log(f"{len(wav_files)} files found.")
    log("")

    if len(wav_files) == 0:
        log("No WAV files found.")
        return None

    for i, wav in enumerate(wav_files, start=1):

        log(f"[{i}/{len(wav_files)}] Processing : {wav.name}")

        try:

            features = extract_all_features(str(wav))

            row = {
                "File": wav.stem
            }

            # Save only scalar values
            for key, value in features.items():

                if isinstance(value, (int, float, str, bool)):
                    row[key] = value

                elif np.isscalar(value):
                    row[key] = value

                # Skip arrays used for plotting
                elif isinstance(value, np.ndarray):
                    continue

            rows.append(row)

            log("    ✓ Done")

        except Exception as e:

            log(f"    ✗ ERROR : {e}")

    df = pd.DataFrame(rows)

    Path(output_csv).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False
    )

    log("")
    log("=" * 60)
    log("Dataset generation completed.")
    log(f"Saved to : {output_csv}")
    log("=" * 60)

    return df