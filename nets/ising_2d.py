import tensorflow as tf
from layers import l_ops
from _ising_config import *



def fully_connected(state: tf.Tensor, fc_size, training=False):
    """
    Makes a binary inference with a single hidden layer
    :param state:
    :param fc_size:
    :param training:
    :return:
    """
    flat = l_ops.flatten_batch(state)
    config = IsingConfig()
    config.is_training = training
    config.weight_decay = 0.001

    with tf.variable_scope('fc'):
        fc_layer = l_ops.fully_connected_layer(
            flat, fc_size, config, activation='RELU')
    with tf.variable_scope('out'):
        out = l_ops.fully_connected_layer(fc_layer, 2, config,
                                          activation='NONE')

    return out


class FerromagnetFullyConnectedNN:
    """
    A simple neural network to classify the phases of the Ising ferromagnet

    [Batch, L, L]

    """
    def __init__(self, fc_size, name='AltCNN'):
        self._config = IsingConfig()
        self._config.weight_decay = None

        self._name = name
        with tf.variable_scope(self._name):
            self._fc_layer = l_ops.FullyConnectedLayer()