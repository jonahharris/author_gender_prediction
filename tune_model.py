"""Module to demonstrate hyper-parameter tuning.
Trains n-gram model with different combination of hyper-parameters and finds
the one that works best.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import load_data
import train

FLAGS = None


def tune_ngram_model(data):
    """Tunes n-gram model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels.
    """
    # Select parameter values to try.
    num_layers = [1, 2, 3]
    num_units = [8, 16, 32, 64, 128]

    # Save parameter combination and results.
    params = {
        'layers': [],
        'units': [],
        'accuracy': [],
    }

    # Iterate over all parameter combinations.
    for layers in num_layers:
        for units in num_units:
                params['layers'].append(layers)
                params['units'].append(units)

                accuracy, _ = train.train_ngram_model(
                    data=data,
                    layers=layers,
                    units=units)
                print(('Accuracy: {accuracy}, Parameters: (layers={layers}, '
                       'units={units})').format(accuracy=accuracy,
                                                layers=layers,
                                                units=units))
                params['accuracy'].append(accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='training_data',
                        help='training data directory')
    FLAGS, unparsed = parser.parse_known_args()

    args = parser.parse_args()

    data_path = args.data_dir

    # Tune model
    train_loader = load_data.TrainDataLoader(data_path)
    data = train_loader.prepare_train_data()

    tune_ngram_model(data)