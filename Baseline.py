import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random

random.seed(0)

from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt

from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
import seaborn as sns
from scipy.stats import kstest

from pandas_profiling import ProfileReport

profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_notebook_iframe()


def explain(pipe, X):

    explainer = shap.Explainer(pipe.named_steps["model"])
    xx = pd.DataFrame(pipe[:-1].transform(X), columns=X.columns)
    shap_values = explainer(xx)

    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)


data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
subm = pd.read_csv("data/sample_submission.csv")

data["train"] = 1
test["train"] = 0

df = data.append(test)

data = df[df["train"] == 1]
test = df[df["train"] == 0]

X_tr, X_te, y_tr, y_te = train_test_split(
    data.drop(columns=["train", "output_gen"]), data["output_gen"], random_state=0
)


imp = SimpleImputer(strategy="mean")
# model = CatBoostRegressor(verbose=0,iterations=200)
model = Lasso()
pipe = Pipeline([("imputer", imp), ("model", model)])

pipe.fit(X_tr, y_tr)

mean_squared_error(pipe.predict(X_te), y_te, squared=False)

imp = SimpleImputer()
model = CatBoostRegressor(verbose=0, iterations=200)

pipe = Pipeline([("imputer", imp), ("model", model)])

pipe.fit(X_tr, y_tr)

mean_squared_error(pipe.predict(X_te), y_te, squared=False)
