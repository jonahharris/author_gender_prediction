"""Module to vectorize data.
Converts the given training and validation texts into numerical tensors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pickle

from textblob import Word
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorization parameters

# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this will be truncated.
MAX_SEQUENCE_LENGTH = 500

# Customize stopwords
stop_words = set(stopwords.words('english'))
stop_words.remove('he')
stop_words.remove('she')
stop_words.remove('his')
stop_words.remove('her')
stop_words.remove('him')
stop_words.remove('himself')
stop_words.remove('herself')
stop_words.remove('i')
stop_words.remove('you')
stop_words.remove('your')
stop_words.remove('we')
stop_words.remove('my')
stop_words.remove('me')
stop_words.remove('on')
stop_words.remove('off')
stop_words.remove('have')
stop_words.remove('do')
stop_words.remove('don')
stop_words.remove('myself')
stop_words.remove('being')
stop_words.remove('to')
stop_words.add('u')
stop_words.add("'")
stop_words.add(u"’")

def custom_tokenizer(sentence):
    word_tokens = word_tokenize(sentence) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #words = [Word(word).lemmatize() for word in filtered_sentence]
    words = filtered_sentence
    return words

def ngram_vectorize(train_texts, val_texts):
    """Vectorizes texts as ngram vectors.
    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.
    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.
    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
            'stop_words': None,
            'max_features': TOP_K,
            'tokenizer': custom_tokenizer,
            'use_idf': True,
            'max_df': 0.95
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    #save vectorizer
    with open("models/tfidf_vectorizer.pickle", 'wb') as fin:
        pickle.dump(vectorizer, fin)


    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    return x_train, x_val

