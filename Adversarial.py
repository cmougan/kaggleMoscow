import pandas as pd
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from catboost import CatBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
subm = pd.read_csv("data/sample_submission.csv")


data.output_gen = 0
test["output_gen"] = 1

df = data.append(test)


df = df.drop(columns=["id"])


X = df.drop(columns="output_gen")
y = df["output_gen"]

enc = TargetEncoder()
X = enc.fit_transform(X, y)

X_tr, X_te, y_tr, y_te = train_test_split(X, y)

cb = CatBoostClassifier(verbose=0, iterations=200)
cb.fit(X_tr, y_tr)

roc_auc_score(y_te, cb.predict(X_te))
