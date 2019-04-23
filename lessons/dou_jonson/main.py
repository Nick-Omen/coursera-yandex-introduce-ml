import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from utils import save_answer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
train = pd.read_csv(os.path.join(BASE_DIR, 'close_prices.csv'))
test = pd.read_csv(os.path.join(BASE_DIR, 'djia_index.csv'))


def run():
    clf = PCA(n_components=10)
    clf.fit(train.values[:, 1:])

    total_dispersion = 0.0
    dispersion_components_gt_90_enough = 0
    for r in clf.explained_variance_ratio_:
        total_dispersion += r
        dispersion_components_gt_90_enough += 1
        if total_dispersion >= 0.9:
            break
    print('Components enough fot 90% dispersion:',  dispersion_components_gt_90_enough)
    save_answer(os.path.join(BASE_DIR, '1.txt'), str(dispersion_components_gt_90_enough))

    transformed_train = clf.transform(train.values[:, 1:])
    X = transformed_train[:, 0]
    corr_coef = np.corrcoef(X, test['^DJI'])[0, 1]
    print('Pirson correlation coef:',  corr_coef)
    save_answer(os.path.join(BASE_DIR, '2.txt'), str(round(corr_coef, 2)))

    first_component_list = list(clf.components_[0])
    company_max_weight_index = first_component_list.index(max(first_component_list))
    company_max_weight = list(train.columns)[company_max_weight_index + 1]
    print('Company with max weight on first component:', company_max_weight)
    save_answer(os.path.join(BASE_DIR, '3.txt'), company_max_weight)
