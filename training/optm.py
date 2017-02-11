import tensorflow as tf
from _ising_config import *


def loss_function(logits, labels):
    # from tensorflow cifar10 example
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def loss_summary(total_loss):
    """
      Args:
        total_loss: Total loss from loss().
      Returns:
        loss_averages_op: op for generating moving averages of losses.
      """

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    with tf.name_scope('losses'):
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l, collections=['LOSSES'])
            tf.summary.scalar(l.op.name, loss_averages.average(l), collections=['LOSSES'])

    return loss_averages_op


def exp_decay_grad_config(init_rate, decay_rate, steps_per_decay, global_step):
    lr = tf.train.exponential_decay(init_rate, global_step,
                                    steps_per_decay, decay_rate, staircase=True)
    config = TrainingConfig('GRAD', lr)
    tf.summary.scalar('learning_rate', lr)
    return config


def make_train_op(total_loss, global_step, hparams: dict):
    """
      Create an optimizer and apply to all trainable variables. Add moving
      average for all trainable variables.
      Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
        train_config:
      Returns:
        train_op: op for training.
      """
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = loss_summary(total_loss)

    grads = None
    opt = None
    method = hparams.get('train_method', 'GRAD')
    # Select the training method
    with tf.control_dependencies([loss_averages_op]):
        if method == 'ADAM':
            print("Using Adam Method.")
            opt = tf.train.AdamOptimizer(hparams['learn_rate'])  # default adam optimizer
        elif method == 'GRAD':
            # Compute gradients.
            if 'decay_rate' in hparams:
                learn_rate = tf.train.exponential_decay(
                    hparams['learn_rate'], global_step,
                    hparams['decay_rate'], hparams['decay_steps'],
                    staircase=True)
            else:
                learn_rate = hparams['learn_rate']

            opt = tf.train.GradientDescentOptimizer(learn_rate)

        elif method == 'MOMNT':
            print("Using momentum method")
            learn_rate = hparams['learn_rate']
            momnt = hparams['momentum']
            opt = tf.train.MomentumOptimizer(learn_rate, momnt)
        else:
            raise ValueError("Unknown training method")
        grads = opt.compute_gradients(total_loss)


    # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #   tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    with tf.name_scope('grads'):
        with tf.name_scope('clipping'):
            # Cap the gradients to avoid overflow crashes
            capped_grads = \
                [
                    (tf.clip_by_value(gv[0], -100.0, 100.0)
                     if gv[0] is not None else None,
                     gv[1])
                    for gv in grads]

        # Apply gradients. global_step incremented by 1
        apply_gradient_op = opt.apply_gradients(capped_grads, global_step=global_step)
        for grad, var in capped_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name, grad, collections=['GRADIENTS'])

    # Track the moving averages of all trainable variables.
    #variable_averages = tf.train.ExponentialMovingAverage(
     #   0.9999, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # train_op executes these two ops
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op
