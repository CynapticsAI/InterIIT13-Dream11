import pandas as pd
import numpy as np
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import AdaBoostClassifier


def train_test_model(train_df, test_df):
    """
    This model requires the input of the train df and the test df
    and will train the model (Use the model UI for the training and testing of the model)
    """
    train_df.pop('date')
    test_df.pop('date')

    feature_columns = list(train_df.columns[:7]) + list(train_df.columns[359:])
    target_columns = list(train_df.columns[7:359])
    fanatasy_columns = target_columns[36::15]
    X_train = train_df[feature_columns]
    y_train = train_df[fanatasy_columns]
    X_test = test_df[feature_columns]
    y_test = test_df[fanatasy_columns]

    y_real = y_test.to_numpy()

    estimator = AdaBoostClassifier(n_estimators=10)
    model = MultiOutputRegressor(estimator)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    n = len(y_pred)
    mae_temp = []
    for i in range(n):
        sorted_indices = np.argsort(y_pred[i])[::-1]
        sum1 = 0
        sum2 = 0

        for j in range(11):
            sum2 += y_real[i][sorted_indices[j]]

        np.sort(y_real[i])[::-1]

        for j in range(11):
            sum1 += y_real[i][j]

        mae_temp.append(abs(sum1 - sum2))

    mae = np.mean(mae_temp)

    with open('../model artifacts/model.pkl', 'wb') as f:
        pickle.dumps(model, f)

    return mae
