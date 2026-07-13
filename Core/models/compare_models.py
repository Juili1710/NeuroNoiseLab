"""
Compare different ML models
Currently supports:
    • Multiple Linear Regression
    • Artificial Neural Network
    • Random Forest
    • SVM
    • XGBoost
"""

import pandas as pd
import matplotlib.pyplot as plt

from Core.models.MLR import train_multiple_linear_regression
from Core.models.ann import train_ann


def compare_models(

        csv_file,

        input_features,

        target,

        validation="LOOCV"

):
    """
    Compare multiple regression models.

    Parameters
    ----------
    csv_file : str

    input_features : list

    target : str

    validation : str

    Returns
    -------
    comparison_df : pandas.DataFrame
    """

    #######################################################
    # Train MLR
    #######################################################

    mlr = train_multiple_linear_regression(

        csv_file=csv_file,

        input_features=input_features,

        target=target,

        validation=validation

    )

    #######################################################
    # Train ANN
    #######################################################

    ann = train_ann(

        csv_file=csv_file,

        input_features=input_features,

        target=target,

        validation=validation

    )

    #######################################################
    # Comparison table
    #######################################################

    comparison_df = pd.DataFrame({

        "Model":[

            "Multiple Linear Regression",

            "Artificial Neural Network"

        ],

        "Validation":[

            validation,

            validation

        ],

        "R2":[

            mlr["R2"],

            ann["R2"]

        ],

        "RMSE":[

            mlr["RMSE"],

            ann["RMSE"]

        ],

        "MAE":[

            mlr["MAE"],

            ann["MAE"]

        ]

    })
    # -------------------------------------------------------
# Rank models (Higher R2 is better, Lower RMSE/MAE better)
# -------------------------------------------------------

    comparison_df["Score"] = (
        comparison_df["R2"].rank(ascending=False)
        + comparison_df["RMSE"].rank(ascending=True)
        + comparison_df["MAE"].rank(ascending=True)
    )

    comparison_df["Rank"] = comparison_df["Score"].rank(method="dense").astype(int)

    comparison_df = comparison_df.sort_values("Rank")

    comparison_df = comparison_df.drop(columns=["Score"])

    #######################################################
    # Save comparison
    #######################################################

    comparison_df.to_csv(

        "Data/metadata/model_comparison.csv",

        index=False

    )

    #######################################################
    # Print
    #######################################################

    print()

    print("="*70)

    print("MODEL COMPARISON")

    print("="*70)

    print()

    print(comparison_df)

    #######################################################
    # Plot
    #######################################################

    plot_model_comparison(comparison_df)

    return comparison_df


###############################################################
# Plot comparison
###############################################################

def plot_model_comparison(df):

    metrics = ["R2","RMSE","MAE"]

    for metric in metrics:

        plt.figure(figsize=(6,4))

        plt.bar(

            df["Model"],

            df[metric]

        )

        plt.ylabel(metric)

        plt.title(f"Model Comparison : {metric}")

        plt.tight_layout()

        plt.savefig(

            f"Results/{metric.lower()}_comparison.png",

            dpi=300

        )

        plt.show()