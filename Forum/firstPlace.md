# Iterative regression imputation [Public 1st place]
https://www.kaggle.com/c/now-you-are-playing-with-power/discussion/300748

Handling missing values was very important in this competition.

We used iterative regression imputation (LightGBM) Implementation to impute missing values, and the performance was significantly improved.

Assume you have N features x1, x2, …, xN.
If i-th (1 <= i <= n) feature has missing values, create a regression (classification) model to predict missing values from all other features. Then perform this for all the features with missing values.

If missing data occur not completely at random, "missing" itself should have information. Thus, adding missing flags for some features improved model performance as well.