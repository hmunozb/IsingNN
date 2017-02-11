import tensorflow as tf
from layers import l_ops
from _ising_config import *
from layers import prepr
from layers import util
import numpy as np


def cnn1(state: tf.Tensor, training=False):
    per_state = prepr.periodicize_3d_batch(state, 2)
    config = IsingConfig()
    config.is_training = training
    config.weight_decay = None
    with tf.variable_scope('conv1'):
        conv_layer = l_ops.conv_3d_layer(
            per_state, 64, config, pool='AVG')
    with tf.variable_scope('fc1'):
        flt = l_ops.flatten_batch(conv_layer)
        fc_layer = l_ops.fully_connected_layer(flt, 64, config)
        drp = tf.nn.dropout(fc_layer, 0.5)
    with tf.variable_scope('out'):
        out_layer = l_ops.fully_connected_layer(
            drp, 2, config, activation='NONE')
    return out_layer


def isg_q_states(state: tf.Tensor):
    """

    :param state: batch  x L^n x C tensor, where C is even
    :return: overlap tensor of dimension batch x C/2 x L^2
    """
    with tf.name_scope('isg_qi'):
        rnk = len(state.get_shape().as_list())
        splt0, splt1 = tf.split(rnk - 1, 2, state)
        out = splt0 * splt1

    return out


class SGNNBase:
    def evaluate(self, tensor: tf.Tensor, training=False):
        pass


class MultiInstanceConv3D(SGNNBase):
    """
    Neural Network candidate for classifying by spin glass phase in the
    3D EA model
    The evaluation input is a tensor of shape
        [batch, N, 2R, L, L, L]
    where L is the model length, N is the number of instances sampled from
    the same point in the phase space, and 2R is the number of replicas sampled
    per instance. This is a tensor of magnetic states, and the 2R replica states
    are reduced to R by calculating replica overlap configurations pairwise.

    Let all convolutions have 2^3 kernels with max pooling and periodic boundary
    conditions.
    [batch, N, 2R, L, L, L]
        -> Expand last dim
            [batch, N, 2R, L, L, L, 1]
        -> Reduce into overlap configurations pairwise
            [batch, N, R, L, L, L, 1]
        -> Map over batch
                [N, R, L, L, L, 1]
                    -> Map over instances
                        [R, L, L, L, 1]
                            -> 3D Convolutional Layer (16 channels, RELU)
                                [R, l, l, l, 16]
                            -> 3D Convolutional Layer (16 channels, RELU)
                                [R, l', l', l', 16]
                            -> Flatten tail dimensions
                                [R, 16*l^3]
                            -> Fully Connected (NONE)
                                [R, 1024]
            [batch, N, R, 1024]
        -> Average over replicas, then RELU activation
            [batch, N, 1024]
        -> Fully connected layer (NONE)
            [batch, N, 512]
        -> Average over instances, then RELU activation
            [batch, 512]
        -> Fully connected layer (NONE)
    [batch, 3]

    """
    def __init__(self, name='MIC3'):
        self._config = IsingConfig()
        self._config.weight_decay = 0.001
        self._name = name

        with tf.variable_scope(self._name):
            self._convs = [
                l_ops.Conv3DLayer(
                    1, 32, self._config, name='Conv1', activation='ELU',
                    pooling_type='MAX', kern_cfg=[2, 1],
                    init_bias=0.2, init_w_sd=0.1),
                l_ops.Conv3DLayer(
                    32, 16, self._config, name='Conv2', activation='ELU',
                    pooling_type='MAX', kern_cfg=[2, 1],
                    init_bias=0.2, init_w_sd=0.1
                )
            ]
            self._rep_fc = l_ops.FullyConnectedLayer(
                16*8, 512, self._config, activation='NONE', name='fc1',
                init_bias=.1, init_w_sd=0.1)
            #self._inst_fc = l_ops.FullyConnectedLayer(
             #   512, 512, self._config, activation='NONE', name='fc2',
              #  init_bias=10.0)
            self._fc3 = l_ops.FullyConnectedLayer(
                512, 3, self._config, activation='NONE',
                init_bias=None, name='fc3', init_w_sd=0.1)

    def evaluate(self, tensor: tf.Tensor, training=False):
        """

        :param tensor:
        :return:
        """
        with tf.name_scope(self._name):
            tensor = tf.expand_dims(tensor, axis=[-1])

            shape = tensor.get_shape().as_list()
            flt = shape[0:3] #[batch, N, R]
            rest = shape[3:] #[L, L, L]
            new_shape = [flt[0]*flt[1]*flt[2]]+[8,8,8,1]
            tensor = tf.reshape(tensor, new_shape)

            tensor = prepr.periodicize_3d_batch(tensor, 3)
            conv1 = self._convs[0].evaluate(tensor)
            #conv1 = prepr.periodicize_3d_batch(conv1, 1)
            conv2 = self._convs[1].evaluate(conv1)

            flattened = l_ops.flatten_batch(conv2)
            fc1 = self._rep_fc.evaluate(flattened)
            fc1 = tf.reshape(fc1,
                             [flt[0], flt[1]*flt[2], -1])
            fc1 = tf.reduce_mean(fc1, axis=[1])
            fc1 = tf.nn.elu(fc1)

            #fc2 = self._inst_fc.evaluate(fc1)
            #fc2 = tf.reshape(fc2,
             #                [flt[0], flt[1], -1])
            #fc2 = tf.reduce_mean(fc2, axis=[1])
            #fc2 = tf.nn.sigmoid(fc2)

            out = self._fc3.evaluate(fc1)

        return out








_pp = [1.0, 1.0]
_pm = [1.0, -1.0]
_mp = [-1.0, 1.0]
_mm = [-1.0, -1.0]

class SpinGlassQSqCalc:
    def __init__(self, num_replicas, name='SGQSq'):
        self._c = num_replicas
        self._name = name
        #with tf.name_scope(name):


class IsingFilter3DLayer:
    """
    input tensor is shaped [Batch, L, L, L, C]

    filter shape is [2^3, 1, 8]
    """
    _num_channels = 8
    _filters = [  # C x L x L x L
        [   [_pp, _pp], [_pp, _pp] ],  # 000

        [   [_pp, _pp], [_mm, _mm] ],  # 100
        [   [_pp, _mm], [_pp, _mm] ],  # 010
        [   [_pm, _pm], [_pm, _pm] ],  # 001

        [   [_pp, _mm], [_mm, _pp] ],  # 110
        [   [_pm, _pm], [_mp, _mp] ],  # 101
        [   [_pm, _mp], [_pm, _mp] ],  # 011

        [   [_pm, _mp], [_mp, _pm] ],  # 111
    ]

    def __init__(self, input_len,  name="IsingFilter3D"):
        self._config = IsingConfig()
        self._config.weight_decay = None
        self._name = name
        self._np_filters = np.asarray(self._filters)
        # shape filters into correct shape and make
        # the filter tensor
        self._tf_filters = tf.constant(
            np.expand_dims(
                self._np_filters.transpose((1, 2, 3, 0)),
                -2))
        self._ker_stride = [1, 2, 2, 2, 1]

        with tf.variable_scope(self._name):
            self._K_params = util.init_const_weights(
                'Params', [self._num_channels], 1.0, None)
            self._Bias = util.init_biases_2(
                'Bias', self._num_channels, 1.0)
            self._kernel = tf.mul(
                self._tf_filters, self._K_params)

    def evaluate(self, tensor: tf.Tensor):
        with tf.name_scope(self._name):
            tensor = tf.expand_dims(tensor, axis=[-1])
            prepr.periodicize_3d_batch(tensor, 1)
            conv = tf.nn.conv3d(
                tensor, self._kernel,
                self._ker_stride, padding='VALID')

        return conv


class IsingSGRecurrentNN:

    def __init__(self, input_size):
        self._in_sz = input_size
        self._cell = tf.nn.rnn_cell.BasicLSTMCell(16)

    def evaluate(self, tensor: tf.Tensor):
        with tf.name_scope('ISGRNN'):
            outputs, _ = tf.nn.dynamic_rnn(self._cell, tensor)