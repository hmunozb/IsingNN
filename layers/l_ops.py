from _ising_config import *
from .util import *

import tensorflow as tf


def _make_activation(z: tf.Tensor, activation: str, name=None):
    if activation == 'RELU':
        actv = tf.nn.relu(z, name=name)
    elif activation == 'ELU':
        actv = tf.nn.elu(z, name=name)
    elif activation == 'SIGM':
        actv = tf.nn.sigmoid(z, name=name)
    elif activation == 'TANH':
        actv = tf.nn.tanh(z, name=name)
    elif activation == 'NONE':
        actv = tf.identity(z, name=name)
    else:
        print("Warning: ignoring unknown activation ", activation)
        actv = z

    return actv


#  todo: if pooling, automatically pad tensor periodically so
    #  its size is divisible by the pooling result
def _perform_3d_pooling(a: tf.Tensor, len, stride, pool='MAX'):
    """

    :param a: Batch x L^3 x C tensor
    :param len: length of pooling
    :param stride: pooling stride
    :return:
    """
    pool_sz = [1, len, len, len, 1]
    pool_strd = [1, stride, stride, stride, 1]
    if pool == 'MAX':
        out_pool = tf.nn.max_pool3d(a, pool_sz, pool_strd, 'VALID')
    elif pool == 'AVG':
        out_pool = tf.nn.avg_pool3d(a, pool_sz, pool_strd, 'VALID')
    elif pool == 'NONE':
        out_pool = tf.identity(a)
    else:
        raise ValueError("pool is not a known value")

    return out_pool


_default_2d_pooling = [2, 2]
_default_2d_kern = [3, 1]


def conv_2d_layer(in_tensor: tf.Tensor,
                  out_channels: int, config: IsingConfig,
                  pool='MAX',
                  pooling=_default_2d_pooling,
                  kern_cfg = _default_2d_kern,
                  bc = 'VALID'):
    """
    remember to enclose in a variable scope
    :param in_tensor: A batch x L x L x 1 tensor
    :param out_channels:
    :param config:
    :param pool: 'MAX', 'AVG', or 'NONE'
    :param pooling: a tuple [pool length, pool stride]
    :param kern_cfg: a tuple [kernel len, kernel stride]
    :param bc: 'VALID', 'SAME', or 'PERIOD'
    :return:
    """
    in_channels = in_tensor.get_shape().as_list()[3]
    ker_shape = [kern_cfg[0], kern_cfg[0],
                        in_channels, out_channels]
    l_st = kern_cfg[1]
    ker_stride = [1, l_st, l_st, 1]
    ker = init_decay_weights('ker', ker_shape, 0.1,
                             config.weight_decay, config.is_training)
    conv = tf.nn.conv2d(in_tensor, ker, ker_stride, 'VALID')
    bias = init_biases('biases', out_channels,
                       1.0, config.is_training, init='CONST')
    z = tf.nn.bias_add(conv, bias)
    actv = _make_activation(z, 'RELU', 'actv')

    pool_sz = [1, pooling[0], pooling[0], 1]
    pool_strd = [1, pooling[1], pooling[1], 1]
    if pool == 'MAX':
        out_pool = tf.nn.max_pool(actv, pool_sz, pool_strd, 'VALID')
    elif pool == 'AVG':
        out_pool = tf.nn.avg_pool(actv, pool_sz, pool_strd, 'VALID')
    elif pool == 'NONE':
        out_pool = tf.identity(actv)
    else:
        raise ValueError("pool is not a known value")

    return out_pool


_default_3d_pooling = [2, 2]
_default_3d_kern = [2, 1]


def conv_3d_layer(in_tensor: tf.Tensor,
                  out_channels: int, config: IsingConfig,
                  activation='RELU',
                  pool='MAX', pooling=_default_3d_pooling,
                  kern_cfg=_default_3d_kern,
                  bc='VALID'):
    """
    remember to enclose in a variable scope
    :param in_tensor: A batch x L x L x L x C tensor
    :param out_channels:
    :param config:
    :param pool: 'MAX', 'AVG', or 'NONE'
    :param pooling: a tuple [pool length, pool stride]
    :param kern_cfg: a tuple [kernel len, kernel stride]
    :param bc: 'VALID', 'SAME', or 'PERIOD'
    :return:
    """
    in_channels = in_tensor.get_shape().as_list()[4]
    ker_shape = [kern_cfg[0], kern_cfg[0], kern_cfg[0],
                    in_channels, out_channels]
    l_str = kern_cfg[1]
    ker_stride = [1, l_str, l_str, l_str, 1]
    ker = init_decay_weights('ker', ker_shape, 0.1,
                             config.weight_decay, config.is_training)

    conv = tf.nn.conv3d(in_tensor, ker, ker_stride, 'VALID')
    bias = init_biases('biases', out_channels, 1.0, config.is_training)
    z = tf.nn.bias_add(conv, bias)
    actv = _make_activation(z, activation)

    pool_sz = [1, pooling[0], pooling[0], pooling[0], 1]
    pool_strd = [1, pooling[1], pooling[1], pooling[1], 1]
    #  todo: if pooling, automatically pad tensor periodically so
    #  its size is divisible by the pooling result

    # Select pooling method
    if pool == 'MAX':
        out_pool = tf.nn.max_pool3d(actv, pool_sz, pool_strd, 'VALID')
    elif pool == 'AVG':
        out_pool = tf.nn.avg_pool3d(actv, pool_sz, pool_strd, 'VALID')
    elif pool == 'NONE':
        out_pool = tf.identity(actv)
    else:
        raise ValueError("pool is not a known value")

    return out_pool


# Class layers have the advantage of not needing reuse_variables
# Should also make it easier to perform more complex training workflows
class Conv3DLayer:
    def __init__(self, in_channels: int, out_channels: int, config: IsingConfig,
                 name='Convolution3D', activation='RELU', pooling_type='MAX',
                 pool_lens=_default_3d_pooling,
                 kern_cfg=_default_3d_kern, init_bias=1.0, init_w_sd=0.1,
                 bc='VALID'):
        """

        :param in_channels: If None, initialization is delayed
        :param out_channels:
        :param config: global configurations
        :param name: the name string
        :param activation: string 'RELU' 'SIGM' or 'NONE'
        :param pooling_type:  string 'MAX' 'AVG' or 'NONE'
        :param pool_lens: array [pool length, pool stride]
        :param kern_cfg: array [kernel length, kernel stride]
        :param bc: string boundary conditions for the convolution
        """
        self._in_C = in_channels
        self._out_C = out_channels
        self._ker_len = kern_cfg[0]
        self._ker_shape = [self._ker_len, self._ker_len, self._ker_len,
                           self._in_C, self._out_C]
        l_str = kern_cfg[1]
        #todo: assert that arguments are of expected types
        self._ker_stride = [1, l_str, l_str, l_str, 1]
        self._activation = activation
        self._pool_len = pool_lens[0]
        self._pool_stride = pool_lens[1]
        self._pooling = pooling_type
        self._name = name
        with tf.variable_scope(self._name):
            self._Kernel = init_decay_weights_2(
                'Kernel', self._ker_shape, init_w_sd, config.weight_decay)
            self._Bias = init_biases_2('Bias', out_channels, init_bias)

    def _inner_eval(self, tensor: tf.Tensor):
        conv = tf.nn.conv3d(
            tensor, self._Kernel, self._ker_stride, 'VALID')
        z = tf.nn.bias_add(conv, self._Bias)
        actv = _make_activation(z, self._activation)
        pooled = _perform_3d_pooling(
            actv, self._pool_len, self._pool_stride, self._pooling)
        tf.summary.histogram('activation', pooled, collections=['STATS'])
        if self._activation == 'RELU':
            tf.summary.scalar(
                'actv_sparsity', tf.nn.zero_fraction(pooled),
                collections=['STATS'])
        return pooled

    def evaluate(self, tensor: tf.Tensor, scope=True):
        if scope:
            with tf.name_scope(self._name):
                pooled = self._inner_eval(tensor)
        else:
            pooled = self._inner_eval(tensor)

        return pooled


def flatten_batch(in_tensor: tf.Tensor):
    batch_sz = in_tensor.get_shape().as_list()[0]
    flat = tf.reshape(in_tensor, [batch_sz, -1], name='flatten')
    return flat


def fully_connected_layer(in_tensor: tf.Tensor,
                          size, config: IsingConfig,
                          activation = 'RELU'):
    """

    :param in_tensor: flattened tensor, batch x k
    :param size: size of neurons in output layer
    :param config:
    :return: batch x size tensor
    """
    k = in_tensor.get_shape().as_list()[1]
    w_shape = [k, size]
    w = init_decay_weights('weights', w_shape, 0.01,
                           config.weight_decay, config.is_training)
    b = init_biases('biases', size, 1.0, config.is_training)
    z = tf.nn.bias_add(tf.matmul(in_tensor, w), b)
    actv = _make_activation(z, activation)

    return actv


class FullyConnectedLayer:
    def __init__(self, input_size, output_size,
                 config: IsingConfig, name='FullyConnected',
                 activation='RELU', init_bias=2.0, init_w_sd=1.0):
        self._name = name
        self._in = input_size
        self._out = output_size
        self._w_shape = [input_size, output_size]
        self._activation = activation
        with tf.variable_scope(self._name):
            self._W = init_decay_weights_2(
                'Weights', self._w_shape, init_w_sd, config.weight_decay)

            self._Bias = init_biases_2('Bias', self._out, init_bias)

    def _inner_eval(self, tensor: tf.Tensor):
        z = tf.nn.bias_add(tf.matmul(tensor, self._W), self._Bias)

        actv = _make_activation(z, self._activation)
        tf.summary.histogram('activation', actv, collections=['STATS'])
        if self._activation == 'RELU':
            tf.summary.scalar(
                'actv_sparsity', tf.nn.zero_fraction(actv), collections=['STATS'])
        return actv

    def evaluate(self, tensor: tf.Tensor, scoped=True):
        if scoped:
            with tf.name_scope(self._name):
                actv = self._inner_eval(tensor)
        else:
            actv = self._inner_eval(tensor)

        return actv

