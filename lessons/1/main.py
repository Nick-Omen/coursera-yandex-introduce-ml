import os
import pandas as pd

data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic.csv'), index_col='PassengerId')


def calc_number_male_female():
    """
    Какое количество мужчин и женщин ехало на корабле?
    В качестве ответа приведите два числа через пробел.
    """
    data_counts = data['Sex'].value_counts()
    print('Male:', data_counts['male'], 'Female:', data_counts['female'])


def calc_survived_percent():
    """
    Какой части пассажиров удалось выжить?
    Посчитайте долю выживших пассажиров.
    Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
    округлив до двух знаков.
    """
    survived_percent = data['Survived'].mean()
    print('Survived percent:', round(survived_percent * 100, 2))


def calc_first_class_percent():
    """
    Какую долю пассажиры первого класса составляли среди всех пассажиров?
    Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
    округлив до двух знаков.
    """
    cabin_class_counts = data['Pclass'].value_counts()
    print('First class percent:', cabin_class_counts[1] / data['Pclass'].count() * 100)


def calc_age_avg_median():
    """
    Какого возраста были пассажиры?
    Посчитайте среднее и медиану возраста пассажиров.
    В качестве ответа приведите два числа через пробел.
    """
    median_age = data['Age'].median()
    avg_age = data['Age'].mean()
    print('Average age:', round(avg_age, 2), 'Median age:', median_age)


def calc_pirsons_corr():
    """
    Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
    Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
    """
    print('Pirson`s correlation is:', data.corr()['SibSp']['Parch'])


def calc_most_popular_woman_name():
    """
    Какое самое популярное женское имя на корабле?
    Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
    Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
    Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
    Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
    а также разделения их на женские и мужские.
    """
    print(data['Name'].apply(lambda n: n.split(', ')[1].split(' (')[0]).value_counts()[:10])


def main():
    calc_number_male_female()
    calc_survived_percent()
    calc_first_class_percent()
    calc_age_avg_median()
    calc_pirsons_corr()
    calc_most_popular_woman_name()


if __name__ == '__main__':
    main()
