import tensorflow as tf
import logging

import numpy as np
from _global_flags import *
from fileio import IsingFileRecord


def _load_chk_or_init(saver, sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if FLAGS.resume_training:
            print('Starting training from checkpoint.')
            saver.restore(sess, ckpt.model_checkpoint_path)
            return
    print('Initializing...')
    sess.run(tf.global_variables_initializer())


class TestResult:
    def __init__(self, batch_t_correct: np.ndarray,
                 batch_count_below_tc: np.ndarray,
                 batch_totals: np.ndarray,
                 config: GlobalConfig):
        """

        :param batch_t_correct: number of examples classifed correctly by temperature
        :param batch_count_below_tc: number of examples classified below tc
        :param batch_totals: total number of examples evaluated at each temperature
        :param config: GlobalConfig
        """
        self.t_correct = batch_t_correct
        self.counts_below_tc = batch_count_below_tc
        self.total = config.eval_steps*config.batch_size

        self.overall_accuracy = float(np.sum(self.t_correct)) / self.total
        self.t_accuracies = self.t_correct.astype(float) / batch_totals
        self.pct_below_tc = self.counts_below_tc.astype(float) / batch_totals


def _eval_once(test_op, batch: IsingFileRecord.IsingBatch,
               config: GlobalConfig, sess: tf.Session):
    """

    :param correct_predict_op:
    :param batch:
    :param config:
    :param num_temps:
    :param sess:
    :return:
    """
    correct_predict_op = tf.cast(tf.nn.in_top_k(test_op, batch.label, 1), dtype=tf.int32)
    _, top_predict_op = tf.nn.top_k(test_op, 1)
    # number of correct predictions by batch and T

    batch_t_correct = np.zeros(shape=config.num_temps)
    # number of T < T_C predictions by batch and T
    count_t_less_TC = np.zeros(shape=config.num_temps)
    # total number of inputs by T, since they are shuffled
    totals = np.zeros(shape=config.num_temps)

    for i in range(config.eval_steps):
        correct_batch, predict_batch, label_batch, index_batch = \
            sess.run([correct_predict_op, top_predict_op,
                      batch.label, batch.index])
        for j in range(config.batch_size):
            index = index_batch[j]
            # if predict_batch[j, 0] == label_batch[j]:
            #    batch_t_correct[i, index] += 1
            batch_t_correct[index] += correct_batch[j]
            count_t_less_TC[index] += predict_batch[j, 0]
            totals[index] += 1

    #print(batch_t_correct)
    #print(count_t_less_TC)
    #print(totals)
    result = TestResult(batch_t_correct, count_t_less_TC, totals, config)
    logging.info("Accuracies:\n"+np.array2string(
            result.t_accuracies, 30, 4, separator='\t'))
    return result


def make_global_step():
    return tf.Variable(0, trainable=False, name='global_step')


def train_and_test_model(training_op: tf.Operation,
                         test_tensor: tf.Tensor,
                         global_step: tf.Tensor,
                         batch: IsingFileRecord.IsingBatch,
                         config: GlobalConfig):
    """
    Remember to scope under a default graph
    :param training_op: Runs a training step in the graph.
    Can itself be a no-op
    :param test_tensor: 2D tensor, the direct output layer of the NN
    :param batch: the IsingBatch tensor fed into the *TEST* tensor
    :param config: GlobalConfig
    :return:
    """
    #EVAL_FREQUENCY = int(0.05*FLAGS.max_steps)
    #g = tf.Graph()
    #with g.as_default():
    #global_training_step = tf.Variable(
     #   0, trainable=False, name='global_step')
    ch1 = logging.StreamHandler()
    ch2 = logging.FileHandler('run.log')
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=logging.DEBUG,
        handlers=[ch1, ch2])

    last_eval_accuracy = tf.Variable(
        0, trainable=False, name='eval_acc', dtype=tf.float32)
    next_acc = tf.placeholder(tf.float32, [])
    update_acc = last_eval_accuracy.assign(next_acc)
    acc_sum_op = tf.summary.scalar('Eval_Accuracy', last_eval_accuracy)

    chk_saver = tf.train.Saver()
    sum_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                       tf.get_default_graph())
    sum_writer.add_session_log(
        tf.SessionLog(status=tf.SessionLog.START))
    merged_sum = tf.summary.merge_all()
    sess = tf.Session()
    _load_chk_or_init(chk_saver, sess)
    log_freq = config.max_steps // 20
    if config.log_freq is not None:
        log_freq = config.log_freq
    acc = 0.0

    with sess.as_default():
        #initialization
        coord = tf.train.Coordinator()  # start coordinator and threads for queues
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #training loop
        for step in range(config.max_steps):
            i = global_step.eval() #the current step
            # evaluate
            if i % config.eval_freq == 0:
                eval_result = _eval_once(
                    test_tensor, batch,
                    config, sess)
                acc = eval_result.overall_accuracy
                # update accuracy tensor
                sess.run(update_acc,
                         feed_dict={next_acc: acc})
                # run training and summary
            _, summary = sess.run([training_op, merged_sum])
            #sess.run(training_op)
            if i % config.sum_save_freq == 0:
                sum_writer.add_summary(summary, i)
            #log every so often
            if (step) % log_freq == 0:
                logging.info('Step %d: %f', i, last_eval_accuracy.eval())
                #print("Step ", i, ": ", last_eval_accuracy.eval())
                chk_saver.save(sess, FLAGS.train_dir + "/chk")
        #final evaluation
        eval_result = _eval_once(test_tensor, batch, config, sess)
        sess.run(update_acc,
                 feed_dict={next_acc: eval_result.overall_accuracy})
        acc_sum = sess.run(acc_sum_op)
        sum_writer.add_summary(acc_sum, config.max_steps)
        #final checkpoint
        chk_saver.save(sess, FLAGS.train_dir + "/chk")
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    sum_writer.add_session_log(
        tf.SessionLog(status=tf.SessionLog.STOP))
    sum_writer.flush()
    sum_writer.close()
    sess.close()

    return eval_result

