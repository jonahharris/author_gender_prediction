"""Module to create model.
Helper functions to create a multi-layer perceptron model. These functions take the model hyper-parameters as input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


def mlp_model(layers, units, dropout_rate, input_shape):
    """Creates an instance of a multi-layer perceptron model.
    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
    # Returns
        An MLP model instance.
    """
    op_units = 1
    op_activation = 'sigmoid'
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model