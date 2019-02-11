import pandas as pd
import pickle
import argparse

from tensorflow import keras

import load_data


class PostClassificator:

    def __init__(self, model_file, post_text):
        self.model_file = model_file
        self.post_text = post_text
        self.vectorizer = self.__load_vectorizer()
        self.model = self.__load_model()

    def __text_preprocessing(self):
        processed_text = load_data.base_preprocessing(pd.Series(self.post_text))
        return processed_text

    def __load_vectorizer(self):
        with open("models/tfidf_vectorizer.pickle", 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer

    def __load_model(self):
        model = keras.models.load_model(model_file)
        return model

    def predict(self):
        text_for_classification = self.__text_preprocessing()
        X = self.vectorizer.transform(text_for_classification.values)
        X = X.astype('float32')
        result = self.model.predict(X)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='models/post_mlp_model.h5', help='model file')
    parser.add_argument('--post_text', type=str, default='Hello world!', help='post to classify')
    FLAGS, unparsed = parser.parse_known_args()

    args = parser.parse_args()

    model_file = args.model_file
    post_text = args.post_text

    # Train model
    clf = PostClassificator(model_file, post_text)
    result = clf.predict()

    if result[0][0] < 0.5:
        print("Female")
    else:
        print("Male")