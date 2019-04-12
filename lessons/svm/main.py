import os
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svm-data.csv'))


def run():
    cls = SVC(C=100000, kernel='linear', random_state=241)
    X = data[['B', 'C']]
    y = data['A']
    cls.fit(X, y)
    print('Result vector numbers: ', [v + 1 for v in cls.support_])
