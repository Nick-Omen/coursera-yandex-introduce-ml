import os
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib
from utils import save_answer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DUMP_PATH = os.path.join(BASE_DIR, 'clf.joblib')
ANSWER_PATH = os.path.join(BASE_DIR, 'anwser.txt')
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)


def initialize_classifier(X, y) -> GridSearchCV:
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X, y)
    joblib.dump(gs, MODEL_DUMP_PATH)
    return gs


def load_classifier() -> GridSearchCV:
    return joblib.load(MODEL_DUMP_PATH)


def find_best_C(X, y):
    if os.path.exists(MODEL_DUMP_PATH):
        clf = load_classifier()
    else:
        clf = initialize_classifier(X, y)
    print(clf.cv_results_.__getitem__('params'))
    print(clf.cv_results_.__getitem__('mean_test_score'))


def run():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target
    clf = SVC(kernel='linear', random_state=241)
    clf.fit(X, y)
    names = vectorizer.get_feature_names()
    arr = clf.coef_.toarray()
    arr[0] = [abs(v) for v in arr[0]]
    sorted_weights = arr[::, arr[0, :].argsort()[::-1]]
    top_10_weights = sorted_weights[0, :10]
    words = list()

    for w in top_10_weights:
        index = np.where(arr == w)
        word_index = index[1][0]
        words.append(names[word_index])

    words.sort()
    print('Most weight words are:', words)
    save_answer(os.path.join(BASE_DIR, 'answer.txt'), ','.join(words))
