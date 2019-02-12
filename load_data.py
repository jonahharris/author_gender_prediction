"""Module to load train.
Load train data and make preprocessing.
"""

import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split


def base_preprocessing(series):
    series = series.fillna('')
    emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F"
      u"\U0001F300-\U0001F5FF"
      u"\U0001F680-\U0001F6FF"
      u"\U0001F1E0-\U0001F1FF"
      "]", flags=re.UNICODE)
    for ndx, member in enumerate(series):
      series[ndx] = re.sub(emoji_pattern, lambda m: (' ' + unicodedata.name(m.group()).replace(' ', '') + ' '), series[ndx])
    filter_pattern = r'[\\\\!"#$%&()*+,-./:;<=>?@[\]^_`{|}~®©™\t\n\'<> ]+'
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

