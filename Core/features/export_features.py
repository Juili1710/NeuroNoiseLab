from pathlib import Path
import pandas as pd

from Core.features.psychoacoustics import extract_all_features

def export_dataset_features(folder):

    rows = []

    files = sorted(
        Path(folder).glob("*.wav")
    )

    for file in files:

        print(f"Processing {file.name}")

        features = extract_all_features(str(file))

        row = {

            "stimulus": file.stem

        }

        row.update(features)

        rows.append(row)

    df = pd.DataFrame(rows)

    output = "Data/metadata/objective_features.csv"

    df.to_csv(
        output,
        index=False
    )

    print(df)

    print(
        f"Saved to {output}"
    )