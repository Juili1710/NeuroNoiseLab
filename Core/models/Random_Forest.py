from sklearn.ensemble import RandomForestRegressor

from Core.models import validation


from Core.models.evaluator import evaluate_model

model = RandomForestRegressor(

    n_estimators=100,

    random_state=42

)

return evaluate_model(model=model,

        csv_file=csv_file,

        input_features=input_features,

        target=target,

        validation=validation)