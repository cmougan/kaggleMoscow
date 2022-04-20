# Magic we find - Private 1st
https://www.kaggle.com/c/now-you-are-playing-with-power/discussion/300742

Magic we find : augmentation.

NaN in this dataset seems like synthetic, each feature has about 11% NaN.

So we randomly change some value of each feature to NaN to get more training data.
The augment code is here:

x_aug = x_train.copy()
for col in feature_list:
    x_aug[col] = x_aug[col].sample(frac=0.89)
x_train = pd.concat([x_train, x_aug])
y_train = pd.concat([y_train, y_train])
We repeat the augment 20 times and get about 1.4 million training data.

This augmentation strategy is a bit like dropout and improves our score significantly.