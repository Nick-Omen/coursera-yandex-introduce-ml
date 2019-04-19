import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from utils import save_answer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
train = pd.read_csv(os.path.join(BASE_DIR, 'salary-train.csv'))
test = pd.read_csv(os.path.join(BASE_DIR, 'salary-test-mini.csv'))


def preprocess_data():
    train['FullDescription'] = train['FullDescription'].replace(
        '[^a-zA-Z0-9]', ' ', regex=True
    ).apply(
        lambda t: t.lower()
    )
    test['FullDescription'] = test['FullDescription'].replace(
        '[^a-zA-Z0-9]', ' ', regex=True
    ).apply(
        lambda t: t.lower()
    )
    train['LocationNormalized'].fillna('nan', inplace=True)
    train['ContractTime'].fillna('nan', inplace=True)
    test['LocationNormalized'].fillna('nan', inplace=True)
    test['ContractTime'].fillna('nan', inplace=True)


def find_tfidf_weights():
    tfidfv = TfidfVectorizer(min_df=5)
    train_weights = tfidfv.fit_transform(train['FullDescription'])
    test_weights = tfidfv.transform(test['FullDescription'])
    return train_weights, test_weights


def get_one_hot_tags():
    enc = DictVectorizer()
    train_cat = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
    test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    return train_cat, test_cat


def run():
    preprocess_data()

    train_weights, test_weights = find_tfidf_weights()
    train_cat, test_cat = get_one_hot_tags()

    X_train = hstack([train_cat, train_weights])
    X_test = hstack([test_cat, test_weights])

    clf = Ridge(random_state=241, alpha=1)
    clf.fit(X_train, train['SalaryNormalized'])

    predictions = clf.predict(X_test)

    print('Predicted salary are:', predictions)

    save_answer(os.path.join(BASE_DIR, 'answer.txt'), ' '.join([str(round(v, 2)) for v in predictions]))
