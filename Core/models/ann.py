from sklearn.neural_network import MLPRegressor

from Core.models.evaluator import evaluate_model


def train_ann(

        csv_file,
        input_features,
        target,
        validation="LOOCV"

):

    model = MLPRegressor(

        hidden_layer_sizes=(5,),      # one hidden layer with 5 neurons

        activation="relu",

        solver="adam",

        learning_rate_init=0.01,

        max_iter=5000,

        random_state=42

    )

    return evaluate_model(

        model=model,

        csv_file=csv_file,

        input_features=input_features,

        target=target,

        validation=validation

    )