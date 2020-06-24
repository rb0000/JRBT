import pandas as pd
import numpy as np
import csv
import xgboost as xgb
from xgboost import XGBClassifier
from scikit.sklearn.model_selection import train_test_split
from scikit.sklearn import metrics

# Thank you for using my sample! For more detailed information on XGBoost, visit
# https://www.datacamp.com/community/tutorials/xgboost-in-python and
# https://towardsdatascience.com/exploring-xgboost-4baf9ace0cf6

# This documentation will take you through the steps to run XGBoost for the first time. Have fun!

# This function grabs the .csv file containing our data. It stores that information in a variable.
# Since xgboost only works on number values (no strings), you will need to encode your data if necessary.


def get_heart_data(file_name):
    dict_hd = pd.read_csv(file_name)
    dict_hd.columns = ["age", "sex", "chest_pain", "resting_bp", "cholestrol", "fasting_bs", "resting_ecg",
                       "max_heartrate", "exercise_ind_angina", "oldpeak", "slope", "major_vessels", "thal", "result"]
    return dict_hd


class Boost():
    dataset = get_heart_data("heart_disease")

    # Here, we define our 'independent' and 'dependent' variables. We are trying to predict the Y value.
    X = dataset.drop(["result"])
    Y = dataset["result"]

    # Now we need determine our training and testing data. This is called "data splicing."
    seed = 42
    test_size = .3
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed)

    model = XGBClassifier()
    model.fit(X_train, Y_train)
    print(model)
