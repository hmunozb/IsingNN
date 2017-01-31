import tensorflow as tf


def periodicize_2d_batch(batch, bnd_len):
    top_slice = tf.slice(batch, [0,0,0,0],[-1, bnd_len, -1, -1])
    top_per_batch = tf.concat(1, [batch, top_slice])
    right_slice = tf.slice(top_per_batch,[0,0,0,0], [-1, -1, bnd_len, -1])
    per_batch = tf.concat(2, [top_per_batch, right_slice])
    return per_batch


def periodicize_3d_batch(batch, bnd_len):
    """
    Pads the batch of ising states with periodic boundary conditions
    :param batch: tensor dims k x L x L x L x C
    :param bnd_len:
    :return: a tensor dims k x (L + bnd_len)^3 x C
    """
    with tf.name_scope('periodc_3d'):
        origin = [0, 0, 0, 0, 0]
        top_slice = tf.slice(batch, origin, [-1, bnd_len, -1, -1, -1])
        top_per_batch = tf.concat(1, [batch, top_slice])
        right_slice = tf.slice(top_per_batch, origin, [-1, -1, bnd_len, -1, -1])
        tr_per_batch = tf.concat(2, [top_per_batch, right_slice])
        front_slice = tf.slice(tr_per_batch, origin, [-1, -1, -1, bnd_len, -1])
        per_batch = tf.concat(3, [tr_per_batch, front_slice])

    return per_batch