"""
Validation strategies for ML models
"""

from sklearn.model_selection import (

    LeaveOneOut,

    KFold,

    train_test_split

)


def get_validation(validation_type="LOOCV",
                   random_state=42):

    validation_type = validation_type.upper()

    if validation_type == "LOOCV":

        return LeaveOneOut()

    elif validation_type == "KFOLD":

        return KFold(

            n_splits=5,

            shuffle=True,

            random_state=random_state

        )

    elif validation_type == "TRAIN_TEST":

        return "TRAIN_TEST"

    else:

        raise ValueError(

            f"Unknown validation method : {validation_type}"

        )