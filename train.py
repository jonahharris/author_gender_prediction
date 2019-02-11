"""Module to train n-gram model.
Vectorizes training and validation texts into n-grams and uses that for
training a n-gram model - a simple multi-layer perceptron model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import build_model
import load_data
import vectorize_data

FLAGS = None


def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=16,
                      dropout_rate=0.2,
                      model_file = 'models/post_mlp_model.h5'):
    """Trains n-gram model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        model_file = file to save model
    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = 2
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts
    x_train, x_val = vectorize_data.ngram_vectorize(train_texts, val_texts)

    # Create model instance.
    model = build_model.mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:])

    # Compile model with learning parameters.
    loss = 'binary_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save(model_file)
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='training_data',
                        help='training data directory')
    parser.add_argument('--model_file', type=str, default='models/post_mlp_model.h5',
                        help='model file')
    FLAGS, unparsed = parser.parse_known_args()

    args = parser.parse_args()

    model_file = args.model_file
    data_path = args.data_dir

    # Train model
    train_loader = load_data.TrainDataLoader(data_path)
    train_data = train_loader.prepare_train_data()

    train_ngram_model(train_data, model_file = model_file)