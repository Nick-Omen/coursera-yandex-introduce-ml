import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, precision_recall_curve
from utils import save_answer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
classifications = pd.read_csv(os.path.join(BASE_DIR, 'classification.csv'))
scores = pd.read_csv(os.path.join(BASE_DIR, 'scores.csv'))
SCORE_HEADERS = (
    'score_logreg',
    'score_svm',
    'score_knn',
    'score_tree',
)


def calc_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    matrix = np.array([[tp, fp],
                       [fn, tn]])
    print('Confusion matrix: ', matrix)
    save_answer(os.path.join(BASE_DIR, '1.txt'), ' '.join([str(v) for v in matrix.flatten()]))


def calc_four_metrics(y_true, y_pred):
    metric_scores = list()

    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy score:', accuracy)
    metric_scores.append(accuracy)

    precision = precision_score(y_true, y_pred)
    print('Precision score:', precision)
    metric_scores.append(precision)

    recall = recall_score(y_true, y_pred)
    print('Recall score:', recall)
    metric_scores.append(recall)

    f1 = f1_score(y_true, y_pred)
    print('F1 score:', f1)
    metric_scores.append(f1)

    save_answer(os.path.join(BASE_DIR, '2.txt'), ' '.join([str(round(v, 2)) for v in metric_scores]))


def calc_max_roc_auc_score(y_true):
    calc_scores_list = list()
    calc_scores = dict()
    for score_key in SCORE_HEADERS:
        s = roc_auc_score(y_true, scores[score_key])
        calc_scores_list.append(s)
        calc_scores[s] = score_key

    print('Max AUC-ROC on table:', calc_scores[max(calc_scores_list)])
    save_answer(os.path.join(BASE_DIR, '3.txt'), calc_scores[max(calc_scores_list)])


def calc_max_precision_on_recall_lt70(y_true):
    score_key_by_max_p = dict()
    max_p_list = list()
    for score_key in SCORE_HEADERS:
        precision, recall, thresholds = precision_recall_curve(y_true, scores[score_key])
        ps = list()
        for i, p in enumerate(precision):
            if recall[i] >= 0.7:
                ps.append(p)
        max_p = max(ps)
        score_key_by_max_p[max_p] = score_key
        max_p_list.append(max_p)
    max_p = max(max_p_list)
    print('Max P is:', max_p, 'Metrics is:', score_key_by_max_p[max_p])
    save_answer(os.path.join(BASE_DIR, '4.txt'), score_key_by_max_p[max_p])


def run():
    y_true = classifications['true']
    y_pred = classifications['pred']

    calc_confusion_matrix(y_true, y_pred)
    calc_four_metrics(y_true, y_pred)

    y_true = scores['true']

    calc_max_roc_auc_score(y_true)
    calc_max_precision_on_recall_lt70(y_true)
