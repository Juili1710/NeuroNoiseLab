from xgboost import XGBRegressor

model = XGBRegressor(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42

)

return evaluate_model(model=model,

        csv_file=csv_file,

        input_features=input_features,

        target=target,

        validation=validation)