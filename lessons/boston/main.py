import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor

data_set = load_boston()
X = scale(data_set.data)
y = data_set.target


def run():
    p_values = np.linspace(1, 10, num=200)
    k_folds = KFold(n_splits=5, shuffle=True, random_state=42)

    ac_scores = {}
    for p in p_values:
        ac_scores[p] = list()

    for train_idxs, test_idxs in k_folds.split(X):
        X_train = X[train_idxs]
        y_train = y[train_idxs]

        X_test = X[test_idxs]
        y_test = y[test_idxs]
        for p in p_values:
            rgsr = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
            rgsr.fit(X_train, y_train)
            score = cross_val_score(rgsr, X_test, y_test, scoring='neg_mean_squared_error')
            ac_scores[p].append(score)

    p_best = 0
    score_best = -10000
    for p in ac_scores.keys():
        max_mean = max(np.vstack(ac_scores[p]).mean(axis=0))
        if max_mean > score_best:
            score_best = max_mean
            p_best = p

    print('p_best: ', p_best, 'score_best: ', score_best)
