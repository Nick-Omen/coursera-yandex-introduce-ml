import os
from sklearn import datasets
import shutil


def my_retrieve(url, dst):
    if url == 'https://ndownloader.figshare.com/files/5975967':
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), '20newsbydate.tar.gz'), dst)


datasets.base.urlretrieve = my_retrieve
news_groups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)


def main():
    print(news_groups.data)
    print(news_groups.target)


if __name__ == '__main__':
    main()
