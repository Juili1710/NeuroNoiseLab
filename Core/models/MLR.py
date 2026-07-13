from sklearn.linear_model import LinearRegression
from Core.models.evaluator import evaluate_model



def train_multiple_linear_regression(
        csv_file,
        input_features,
        target,
        validation="LOOCV"
):

    model = LinearRegression()

    # First evaluate the model
    results = evaluate_model(
        model=model,
        csv_file=csv_file,
        input_features=input_features,
        target=target,
        validation=validation
    )

    # Now get trained pipeline
    pipeline = results["Model"]

    coefficients = pipeline.named_steps["model"].coef_
    intercept = pipeline.named_steps["model"].intercept_

    equation = f"{target} = {intercept:.3f}"

    for c, f in zip(coefficients, input_features):
        equation += f" + ({c:.3f})*{f}"

    results["Equation"] = equation

    return results