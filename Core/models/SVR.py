from sklearn.svm import SVR

model = SVR(

    kernel="rbf",

    C=10,

    gamma="scale"

)

return evaluate_model(model=model,

        csv_file=csv_file,

        input_features=input_features,

        target=target,

        validation=validation)