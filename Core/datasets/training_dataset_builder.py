import pandas as pd


def build_training_dataset(
        objective_csv,
        jury_csv,
        output_csv
):
    """
    Merge objective features with
    averaged jury ratings.
    """

    obj = pd.read_csv(objective_csv)

    jury = pd.read_csv(jury_csv)

    # -------------------------------------
    # Rename questionnaire columns
    # -------------------------------------

    jury = jury.rename(

        columns={

            "Q1": "Noticeability",
            "Q2": "Urgency",
            "Q3": "Detectability",
            "Q4": "Annoyance",
            "Q5": "Pleasantness",
            "Q6": "Naturalness",
            "Q7": "Acceptability"

        }

    )

    # -------------------------------------
    # Average ratings for each stimulus
    # -------------------------------------

    jury = jury.groupby(

        "stimulus",

        as_index=False

    ).mean(

        numeric_only=True

    )

    # -------------------------------------
    # Merge
    # -------------------------------------

    dataset = obj.merge(

        jury,

        left_on="File",

        right_on="stimulus",

        how="inner"

    )

    dataset.drop(

        columns=["stimulus"],

        inplace=True

    )

    dataset.to_csv(

        output_csv,

        index=False

    )

    return dataset