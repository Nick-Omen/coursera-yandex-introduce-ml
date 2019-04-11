import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
train_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.csv'))
test_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.csv'))


def train_perceptron(X, y) -> Perceptron:
    perceptron = Perceptron(random_state=241)
    perceptron.fit(X, y)
    return perceptron


def main():
    train = train_data.values
    test = test_data.values

    X_train = train[:, 1:]
    y_train = train[:, 0]

    X_test = test[:, 1:]
    y_test = test[:, 0]

    perceptron = train_perceptron(X_train, y_train)
    predictions = perceptron.predict(X_test)

    default_ac = accuracy_score(y_test, predictions)

    print('Default accuracy:', default_ac)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    perceptron_scaled = train_perceptron(X_train_scaled, y_train)
    predictions_scaled = perceptron_scaled.predict(X_test_scaled)

    scaled_ac = accuracy_score(y_test, predictions_scaled)

    print('Scaled accuracy:', scaled_ac)

    diff = scaled_ac - default_ac

    print('Difference between default and scaled is:', diff)
    output_file = os.path.join(BASE_DIR, 'answer.txt')
    with open(output_file, 'w') as file:
        file.write(str(round(diff, 3)))
        print('Output written to file `{}`'.format(output_file))


if __name__ == '__main__':
    main()