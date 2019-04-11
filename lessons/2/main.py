import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic.csv'),
                   index_col='PassengerId')


def main():
    tree_data = data[['Survived', 'Pclass', 'Fare', 'Age', 'Sex']].dropna()
    tree_data['Sex'] = tree_data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(tree_data[['Pclass', 'Fare', 'Age', 'Sex']], tree_data['Survived'])
    print('----------------------------------')
    print('Pclass', 'Fare', 'Age', 'Sex')
    print(clf.feature_importances_)
    print('----------------------------------')
    passengers = [
        [1, 100.00, 12, 1],
        [1, 100.00, 12, 0],
        [3, 10.00, 48, 1],
        [3, 10.00, 48, 2],
        [1, 83, 57, 0],
        [1, 83, 57, 1],
        [2, 25, 32, 0],
        [2, 25, 32, 1],
    ]
    for p in passengers:
        print("Passenger Class:", p[0])
        print("Paid for Ticket:", p[1], '$')
        print("Age:", p[2])
        print("Sex:", 'male' if p[3] == 1 else 'female')
        print("Will he alive?", clf.predict([p]))
        print('----------------------------------')


if __name__ == '__main__':
    main()
