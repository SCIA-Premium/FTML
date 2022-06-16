import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split, cross_val_score
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, SGDRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost.sklearn import XGBRegressor

# Loading data
inputs = np.load('../data/regression/inputs.npy')
labels = np.load('../data/regression/labels.npy')

labels = np.ravel(labels)

regression_models = {
    "SVR": SVR(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "ElasticNet": ElasticNet(),
    "BayesianRidge" : BayesianRidge(),
    "KernelRidge" : KernelRidge(),
    "LinearRegression" : LinearRegression(),
    "XGBoost" : XGBRegressor()
}

regression_score = {}

def regression_testing(regressor):
    print(f"Testing {regressor[0]}...")
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    model = regressor[1]
    print("Fitting...")
    model.fit(X_train, y_train)
    print("Predicting...")
    scores = cross_val_score(model, inputs, labels, cv=10, scoring="r2")
    print("Scoring...")
    print("%0.2f r2_score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    score = scores.mean()
    print(f"{regressor[0]} score : {score}")
    regression_score[regressor[0]] = score

for key, value in regression_models.items():
    regression_testing((key, value))
    
for i, regressor in enumerate(regression_score.items()):
    print(f"{regressor[0]} score : {regressor[1]}")
    plt.bar(i, regressor[1], label=regressor[0])
plt.legend()
plt.title("Benchmark of multiple regressor r2_score on given dataset")
plt.show()