
import tensorflow as tf


def init_decay_weights(name, shape, stddev, wd, training=False):
    # From tensorflow cifar10 example
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float32 #if FLAGS.use_fp16 else tf.float32
    init = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    var = tf.get_variable(name, shape,
                          initializer=init, dtype=dtype)
    if (wd is not None) and training:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        print('L2 Loss will be added for ', var.op.name)
    if training:
        tf.summary.histogram(name, var)
    return var


def init_decay_weights_2(name, shape, stddev, wd):
    init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    var = tf.get_variable(name, shape,
                          initializer=init, dtype=tf.float32)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        print('L2 Loss will be added for ', var.op.name)
    tf.summary.histogram(name+'/hist', var)
    return var


def init_const_weights(name, shape, val, wd):
    var = tf.Variable(
        tf.constant(val, dtype=tf.float32, shape=shape),
        name=name)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        print('L2 Loss will be added for ', var.op.name)
    tf.summary.histogram(name+'/hist', var)
    return var


def init_biases(name, len, strength, training=False, init='CONST'):
    dtype = tf.float32  # if FLAGS.use_fp16 else tf.float32
    if init == 'CONST':
        init = tf.constant_initializer(strength, dtype)
    elif init == 'GAUSN':
        init = tf.truncated_normal_initializer(0.0, abs(strength))
    else:
        raise ValueError("Unknown value for init: {}".format(init))
    var = tf.get_variable(name, [len],
                          initializer=init, dtype=dtype)
    if training:
        tf.summary.histogram(name, var)

    return var


def init_biases_2(name, len, strength, init='CONST'):
    if init == 'CONST':
        init = tf.constant_initializer(strength, tf.float32)
    elif init == 'GAUSN':
        init = tf.truncated_normal_initializer(0.0, abs(strength))
    else:
        raise ValueError("Unknown value for init: {}".format(init))
    var = tf.get_variable(name, [len],
                          initializer=init, dtype=tf.float32)
    tf.summary.histogram(name+'/hist', var)

    return var

