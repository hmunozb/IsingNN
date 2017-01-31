import tensorflow as tf


def tf_int32_from_bytes(b_arr, pos):
    """
    :param b_arr: 1D tensor of 8 bit uints
    :param pos: position to get byte from
    :return: int32 tensor at pos from bytes in b_arr
    """
    return \
        tf.reshape(
            tf.bitcast(
                tf.slice(b_arr, [pos], [4]),
                tf.int32),
            [])


def tf_float_from_bytes(b_arr, pos):
    """
    :param b_arr: 1D tensor of 8 bit uints
    :param pos: position to get byte from
    :return: float32 tensor at pos from bytes in b_arr
    """
    return \
        tf.reshape(
            tf.bitcast(
                tf.slice(b_arr, [pos], [4]),
                tf.float32),
            [])