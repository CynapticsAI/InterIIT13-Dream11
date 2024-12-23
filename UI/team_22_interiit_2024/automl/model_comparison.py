import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

if os.path.exists('match_data.csv'):
    data = pd.read_csv('match_data.csv')
else:
    import gdown
    url = 'https://drive.google.com/uc?id=1gN08fmoUgsEhl6x1DbB9FHhAhtfztUkr'
    gdown.download(url, 'match_data.csv', quiet=False)

df_copy = data.copy()
df_copy.pop("Unnamed: 0")
df_copy.pop("date")
feature_columns = list(df_copy.columns[:6]) + list(df_copy.columns[336:])
target_columns = list(df_copy.columns[6:336])
fanatasy_columns = target_columns[14::15]
X = df_copy[feature_columns]
y = df_copy[fanatasy_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, shuffle=False
)
y_real = y_test.to_numpy()

model_estimators = [
    'RF',
    'XGBoost',
    'linear',
    'ada',
    'KNeighborsRegressor'
]

model_mae = {}


def simulate_model(model_name):

    if model_name == 'XGBoost':
        estimator = XGBRegressor(),
    elif model_name == 'RF':
        estimator = RandomForestRegressor(
            n_estimators=10, max_depth=12, n_jobs=-1),
    elif model_name == 'linear':
        estimator = LinearRegression()
    elif model_name == 'ada':
        estimator = AdaBoostClassifier(n_estimators=10),
    elif model_name == 'KNeighborsRegressor':
        estimator = KNeighborsRegressor(
            n_neighbors=5, algorithm='auto', weights='uniform')

    print(model_name)
    if (type(estimator) == tuple):
        model = MultiOutputRegressor(estimator[0])
    else:
        model = MultiOutputRegressor(estimator)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    n = len(y_pred)
    a = []
    for i in range(n):
        sorted_indices = np.argsort(y_pred[i])[::-1]
        sum1 = 0
        sum2 = 0

        for j in range(11):
            sum2 += y_real[i][sorted_indices[j]]

        np.sort(y_real[i])[::-1]

        for j in range(11):
            sum1 += y_real[i][j]

        a.append(abs(sum1 - sum2))

    model_mae[model_name] = np.mean(a)


for model in model_estimators:
    simulate_model(model)

models = model_mae.keys()
mae_values = model_mae.values()

plt.figure(figsize=(8, 6))
plt.bar(models, mae_values, color=['blue'])

plt.xlabel('Models', fontsize=14)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
plt.title('Comparison of MAE Across Models', fontsize=16)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

for i, value in enumerate(mae_values):
    plt.text(i, value + 0.05, f"{value:.2f}", ha='center', fontsize=12)

plt.savefig("model_comparison.png")
plt.show()
