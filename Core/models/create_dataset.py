"""
Prepare dataset for ML

Objective Features +
Mean Jury Ratings
"""

import pandas as pd

# ==========================================================
# Subjective Parameter Mapping
# GUI name  ->  Jury CSV Column
# ==========================================================

QUESTION_MAP = {

    "Detectability": "Q1",

    "Urgency": "Q2",

    "Directionality": "Q3",

    "Annoyance": "Q4",

    "Pleasantness": "Q5",

    "Naturalness": "Q6",

    "Acceptability": "Q7"

}

def prepare_training_dataset(
        objective_csv,
        jury_csv,
        output_csv,
        target_columns=None
):
    """
    Parameters
    ----------
    objective_csv : str

    jury_csv : str

    output_csv : str

    target_columns : list

        Example ["Annoyance"] or ["Annoyance","Detectability"]
    """

    if target_columns is None:
        target_columns = ["Annoyance"]

    # -----------------------------
    # Load data
    # -----------------------------

    objective = pd.read_csv(objective_csv)

    jury = pd.read_csv(jury_csv)
# -------------------------------------------------
# Convert friendly names to actual CSV columns
# -------------------------------------------------

    mapped_targets = []

    for target in target_columns:

        if target in QUESTION_MAP:

            mapped_targets.append(
                QUESTION_MAP[target]
            )

        else:

            mapped_targets.append(target)
    # -----------------------------
    # Mean ratings
    # -----------------------------

    grouped = (

        jury

        .groupby("stimulus")[mapped_targets]

        .mean()

        .reset_index()

    )
        # -----------------------------------------
    # Rename Q1,Q2... back to friendly names
    # -----------------------------------------

    reverse_map = {

        v: k

        for k, v in QUESTION_MAP.items()

    }

    grouped.rename(

        columns=reverse_map,

        inplace=True

    )

    grouped.rename(

        columns={"stimulus": "File"},

        inplace=True

    )

    # -----------------------------
    # Merge
    # -----------------------------

    training = pd.merge(

        objective,

        grouped,

        on="File",

        how="inner"

    )

    training.to_csv(

        output_csv,

        index=False

    )

    print()

    print("="*60)

    print("Training Dataset Created")

    print("="*60)

    print()

    print(training)

    return training