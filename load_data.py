"""Module to load train.
Load train data and make preprocessing.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def base_preprocessing(series):
    series = series.fillna('')
    filter_pattern = r'[\\\\!"#$%&()*+,-./:;<=>?@[\]^_`{|}~®©™\t\n\'<>]'
    series = series.str.replace(filter_pattern, ' ')
    series = series.str.lower()
    return series


class TrainDataLoader:
    def __init__(self, train_data_path):
        self.train_data_path = train_data_path

    def _import_female_train_data(self):
        female_posts = pd.read_csv("{path}/female.txt".format(path=self.train_data_path),
                                   delimiter="\t", header=None, names=["post"])
        female_posts['label'] = 0
        return female_posts

    def _import_male_train_data(self):
        male_posts = pd.read_csv("{path}/male.txt".format(path=self.train_data_path),
                                 delimiter="\t", header=None, names=["post"])
        male_posts['label'] = 1
        return male_posts

    def import_train_data(self):
        male_posts = self._import_male_train_data()
        female_posts = self._import_female_train_data()
        train_data = male_posts.append(female_posts, ignore_index=True)
        train_data['post'] = base_preprocessing(train_data['post'])
        return train_data

    def prepare_train_data(self):
        train_data = self.import_train_data()
        X = train_data['post'].values
        y = train_data['label'].values
        train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=0.20, random_state=1)
        return ((train_texts, train_labels), (val_texts, val_labels))