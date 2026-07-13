import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from Core.models.validation import get_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def evaluate_model(

    model,

    csv_file,

    input_features,

    target,

    validation="LOOCV"

):
    df = pd.read_csv(csv_file)

    X = df[input_features]

    y = df[target]
    pipeline = Pipeline([("scaler", StandardScaler()),("model",model)])
    validator = get_validation(validation)
    predictions = []
    actual = []
    if validation == "LOOCV":

        for train_idx, test_idx in validator.split(X):

            X_train = X.iloc[train_idx]

            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]

            y_test = y.iloc[test_idx]

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            predictions.extend(y_pred)

            actual.extend(y_test)

                ### metrics 
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        mae = mean_absolute_error(actual, predictions)

        r2 = r2_score(actual, predictions)

        pipeline.fit(X, y)
        
    
    results = pd.DataFrame({

        "Actual": actual,

        "Predicted": predictions

    })

    results.to_csv(
        "Data/metadata/mlr_predictions.csv",
        index=False
    )
    return {

        "Model": pipeline,
        #"Equation": equation,
        "Input Features": input_features,
        "Target": target,
        "Validation": validation,
        "Metrics":{
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        }, 
        "Results": results
    }
def plot_actual_vs_predicted(
        actual,
        predicted,
        r2,
        rmse,
        mae,
        model_name="Model",
        save_path=None
):
    # """
    # Creates Actual vs Predicted scatter plot
    # """

    plt.figure(figsize=(6,6))

    # Scatter points
    plt.scatter(
        actual,
        predicted,
        s=80,
        alpha=0.8,
        label="Predictions"
    )

    # Ideal prediction line
    
    minimum= 0
    maximum= 5
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
        label="Ideal Prediction"
    )
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    #plt.title("Multiple Linear Regression\nActual vs Predicted")
    plt.title(f"{model_name}\nActual vs Predicted")
    plt.legend()

    # Display performance metrics
    text = (
        f"R² = {r2:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.3)
    )

    plt.tight_layout()
    # Create folder automatically if it doesn't exist
    
    #save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path is None:

        save_path = f"Results/{model_name.replace(' ','_').lower()}_actual_vs_predicted.png"

    save_path = Path(save_path)

    
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.show()