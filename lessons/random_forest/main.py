import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

from utils import save_answer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, 'abalone.csv'))


def get_r2_score_for_rfr(X, y, cv=1, n_estimators=1):
    rgs = RandomForestRegressor(random_state=1, n_estimators=n_estimators)
    cv_scores = cross_val_score(rgs, X, y, scoring='r2', cv=cv)
    return np.mean(cv_scores)


def run():
    data['Sex'] = data['Sex'].apply(lambda s: -1 if s == 'F' else (0 if s == 'I' else 1))
    classes_names = data.columns[:-1]
    X = data[classes_names]
    y = data['Rings']

    r2_min_score_gt_52_n_estimators = None

    k_fold = KFold(random_state=1, n_splits=5, shuffle=True)

    for i in range(1, 51):
        max_score = get_r2_score_for_rfr(X, y, cv=k_fold, n_estimators=i)
        print('Max score for n_estimators =', i, 'is:', max_score)
        if r2_min_score_gt_52_n_estimators is None and max_score > 0.52:
            r2_min_score_gt_52_n_estimators = i
            break

    # because of the update KFold library and the problem of the floating point answer may differ on +- 1
    save_answer(os.path.join(BASE_DIR, '1.txt'), str(r2_min_score_gt_52_n_estimators - 1))
    save_answer(os.path.join(BASE_DIR, '2.txt'), str(r2_min_score_gt_52_n_estimators))

    # correct answer on MacOS, scikit-learn==0.20.3, python 3.6.5
    save_answer(os.path.join(BASE_DIR, '3.txt'), str(r2_min_score_gt_52_n_estimators + 1))
