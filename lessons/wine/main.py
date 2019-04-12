import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wine.csv'))


def get_k_with_acc_on_data(X, y, scaled=False):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    ac_scores = {}

    for k in range(1, 51):
        ac_scores[k] = list()

    for train_idx, test_idx in k_fold.split(X):
        for k in range(1, 51):
            clf = KNeighborsClassifier(n_neighbors=k)
            if scaled:
                train_X = X[train_idx]
                train_y = y[train_idx]
                test_X = X[test_idx]
                test_y = y[test_idx]
            else:
                train_X = X.iloc[train_idx]
                train_y = y.iloc[train_idx]
                test_X = X.iloc[test_idx]
                test_y = y.iloc[test_idx]

            clf.fit(train_X, train_y)

            predictions = clf.predict(test_X)
            accuracy = accuracy_score(test_y, predictions)
            ac_scores[k].append(accuracy)

    k_best = 0
    ac_best = 0.0
    for k in ac_scores.keys():
        mean = np.matrix(ac_scores[k]).mean()
        if mean > ac_best:
            ac_best = mean
            k_best = k
    return k_best, ac_best


def run():
    classes_names = data.columns[1:]
    X = data[classes_names]
    y = data['Class']

    k_best, ac_best = get_k_with_acc_on_data(X, y)
    print('K best: ', k_best, 'Accuracy best: ', round(ac_best, 2))

    k_best, ac_best = get_k_with_acc_on_data(scale(X), y, scaled=True)
    print('K scaled best: ', k_best, 'Accuracy scaled best: ', round(ac_best, 2))
