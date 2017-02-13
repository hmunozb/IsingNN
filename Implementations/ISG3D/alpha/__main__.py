
from _global_flags import *
from nets import sg_3d
from fileio import IsingFileRecord
from training import optm
from training import train_model
from Implementations.ISG3D.MCCNN1 import net

import tensorflow as tf
import numpy as np
import logging

#T_ARR_HI = [2.25,2.2,2.15,2.1,2.05,2.,1.95,1.9,1.85,1.8,1.75,1.7,
#            1.65,1.6,1.55,1.5,1.45,1.4,1.35,1.3,1.25,1.2,1.15,1.1,
 #           1.05,1.]
#T_ARR_LO = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,
 #           0.45,0.4,0.35,0.3]

#T_ARR_1 = T_ARR_HI + T_ARR_LO

T_ARR_HI = [2., 1.82, 1.66, 1.5, 1.34, 1.2, 1.,]

T_ARR_LO = [0.94, 0.82, 0.7, 0.6, 0.5, 0.42, 0.34, 0.27, 0.2]

T_ARR_1 = T_ARR_HI + T_ARR_LO
T_ARR_2 = [1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8,
3., 3.2, 3.4, 3.6, 3.8, 4., 4.2, 4.4,
4.6, 4.8, 5., 5.2, 5.4, 5.6, 5.8, 6.,
6.2, 6.4, 6.6, 6.8, 7., 7.2, 7.4, 7.6]

T_ARR_2.reverse()

TC_FERROMAGNET = 4.511
TC_SPINGLASS = 0.951

PARAMAG_CLASS = 0
FERROMAG_CLASS = 1
SPINGLASS_CLASS = 2


def _label_fn(cls):
    if cls == 0:
        return 0
    else:
        return 1

settings = {
    'L': 8,
    'dim': 3,
    'reps_per_inst': 2,
    'insts_per_exmp': 16,
    'batch_size':   4,

    'max_steps':       500,
    'eval_steps':  16,
    'eval_freq': 50,
    'sum_save_freq': 5,
    'num_temps': len(T_ARR_1),
    't_part': len(T_ARR_HI) - 1,
    'log_freq': 5,
    'enqueue_frac': 0.05  # fraction of epoch size that should be enqueued at all times
}

hparams = {
    'train_method': 'MOMNT',  # 'GRAD', 'MOMNT', 'EXPD', or 'ADAM'
    'learn_rate': 1.0e-3,
    'momentum': 0.9,
    'fc_size':  64,
    'num_in_chans': 2
}


DEFAULT_TRAIN_INPUT_DIR = '/Users/hmunozbauza/Google Drive/C++ Projects/MonteCarlo-0.2/outputs/f2l10_2'
DEFAULT_TEST_OUTPUT_DIR = '/Users/hmunozbauza/Google Drive/C++ Projects/MonteCarlo-0.2/outputs/f2l10_val'


def eval_once_lambd(test_op, batch: IsingFileRecord.IsingBatch,
               config: GlobalConfig, sess: tf.Session):
    pass


class AlphaEval(train_model.TheTrainer):
    def __init__(self, train_directory, config: GlobalConfig,
                        inference_model: sg_3d.SGNNBase):
        super().__init__(train_directory, config)
        self.cnn = inference_model
        self.isfc = IsingFileRecord.IsingFileConfig(settings)
        with tf.name_scope('eval'):
            self.batch = IsingFileRecord.three_phase_input(
                [FLAGS.eval_input_dir],
                [len(T_ARR_1)],
                [10, 0, 1],
                self.isfc)
            logits = self.cnn.evaluate(self.batch.state)



class AlphaTrainer(train_model.TheTrainer):
    def __init__(self, train_directory, config: GlobalConfig,
                 inference_model: sg_3d.SGNNBase):
        super().__init__(train_directory, config)
        self.cnn = inference_model
        alt_cnn_net = self.cnn
        self.isfc = IsingFileRecord.IsingFileConfig(settings)
        with tf.name_scope('training'):
            self.batch = IsingFileRecord.three_phase_input(
                [   FLAGS.input_dir+'/isg',
                    FLAGS.input_dir+'/isf'],
                [len(T_ARR_1), len(T_ARR_2)],
                [[TC_SPINGLASS, SPINGLASS_CLASS, PARAMAG_CLASS],
                 [TC_FERROMAGNET, FERROMAG_CLASS, PARAMAG_CLASS]],
                self.isfc)
            logits = alt_cnn_net.evaluate(self.batch.state, training=True)
            loss = optm.loss_function(logits, self.batch.label)
            self.train = optm.make_train_op(loss, self.step, hparams)
            # image summary

        # tf.get_variable_scope().reuse_variables()
        with tf.name_scope('testing'):
            self.test_batch = IsingFileRecord.three_phase_input(
                [   FLAGS.eval_input_dir + '/isg',
                    FLAGS.eval_input_dir + '/isf'],
                [len(T_ARR_1), len(T_ARR_2)],
                [[TC_SPINGLASS, SPINGLASS_CLASS, PARAMAG_CLASS],
                 [TC_FERROMAGNET, FERROMAG_CLASS, PARAMAG_CLASS]],
                self.isfc)
            self.eval_logits = alt_cnn_net.evaluate(self.test_batch.state, training=False)
        self.summary_op = tf.summary.merge(
            [   tf.summary.merge_all('STATS'), tf.summary.merge_all('TRAINABLE'),
                tf.summary.merge_all('GRADIENTS'), tf.summary.merge_all('LOSSES')])

        self.accuracy_tensor = tf.Variable(
            0, trainable=False, name='eval_acc', dtype=tf.float32)
        self.accuracy_op = tf.identity(self.accuracy_tensor)

        self._init_saver()

        ch1 = logging.StreamHandler()
        ch2 = logging.FileHandler('run.log')
        logging.basicConfig(
            format='%(asctime)s %(message)s',
            level=logging.DEBUG,
            handlers=[ch1, ch2])

    def evaluate(self):
        correct_predict_op = tf.cast(tf.nn.in_top_k(
            self.eval_logits, self.test_batch.label, 1), dtype=tf.int32)
        #softmax = tf.nn.softmax(self.eval_logits)
        _, top_predict_op = tf.nn.top_k(self.eval_logits, 1)
        correct_counts = np.zeros(shape=[2, 2])
        prediction_pcts = np.zeros(shape=[2, 2, 3])
        #totals = [0, 0]
        for i in range(self.config.eval_steps):
            correct, predict, q_i, labels = \
                self.sess.run([correct_predict_op, top_predict_op,
                               self.test_batch.q_index,
                               self.test_batch.label])
            for j in range(self.config.batch_size):
                q_index = q_i[j]
                t_index = _label_fn(labels[j])
                pred_index = predict[j]
                correct_counts[q_index, t_index] += correct[j]
                prediction_pcts[q_index, t_index, pred_index] += 1
                #totals[q_index] += 1
        logging.info(" Predictions:\n"+str(prediction_pcts)
                     +"\n Correct:\n"+str(correct_counts))

    def log(self):
        logging.info('Step %d: %f', self.step.eval(),
                     self.accuracy_op.eval())

    def summarize(self):
        pass

    def train_fn(self):
        self.train_loop(self.train, self.summary_op )


def train():
    global_config = GlobalConfig(settings)
    IsingFileRecord.IsingFileConfig(settings)

    ising_l = FLAGS.ising_l
    epoch_size = FLAGS.epoch_size
    chans = hparams['num_in_chans']
    eval_epoch_size = epoch_size

    if FLAGS.eval_epoch_size > 0:
        eval_epoch_size = FLAGS.eval_epoch_size
    min_q_examples = int(settings['enqueue_frac']*epoch_size)
    if FLAGS.batch_size > 0:
        global_config.batch_size = FLAGS.batch_size

    g = tf.Graph()

    print('Epoch Size: ', epoch_size)
    print('Batch Size: ', global_config.batch_size)
    print('Traning Steps: ', global_config.max_steps)
    with g.as_default():
        #  gradient descent algorithm
        #training_config = optm.grad_desc_config(
         #   hparams['learn_rate'])
        global_step = train_model.make_global_step()
        #training operation
        alt_cnn_net = net.ISGQCNN(hparams['num_in_chans'])
        with tf.name_scope('training'):
            batch = IsingFileRecord.sg_3d_multi_input(
                FLAGS.input_dir, chans, global_config.batch_size,
                min_q_examples, ising_l)
            logits = alt_cnn_net.evaluate(batch.state, training=True)
            loss = optm.loss_function(logits, batch.label)
            train = optm.make_train_op(loss, global_step, hparams)
            # image summary
            with tf.name_scope('img'):
                snap = tf.mul(batch.state[:, 0, ...], batch.state[:, 1, ...])
                snap = tf.reduce_mean(snap, axis=-1)
                snap = tf.expand_dims(snap, axis=-1)
                snap = tf.cast(tf.add(tf.mul(snap, 126), 127), tf.uint8)
                #snap = tf.cast(tf.mul(snap, 126), tf.uint8)
            tf.summary.image('glass', snap, max_outputs=20)

        #tf.get_variable_scope().reuse_variables()
        with tf.name_scope('testing'):
            test_batch = IsingFileRecord.sg_3d_multi_input(
                FLAGS.eval_input_dir, chans, global_config.batch_size,
                min_q_examples, ising_l)
            eval_logits = alt_cnn_net.evaluate(test_batch.state)

        results = train_model.train_and_test_model(
            train, eval_logits, global_step, test_batch, global_config)
    arr = np.array([T_ARR_1, results.pct_below_tc, results.t_accuracies]).transpose()
    print('*** RESULTS *** ')
    for v in arr:
        print("{:.3}: {:.2%} at {:.2%}".format(
            v[0], v[1], v[2]))
    np.savetxt(str(FLAGS.train_dir)+"/Ts.csv", arr, delimiter=',')


def train_2():
    g = tf.Graph()
    config = GlobalConfig(settings)

    with g.as_default():
        cnn = sg_3d.MultiInstanceConv3D()
        tr = AlphaTrainer(FLAGS.train_dir, config, cnn)
        tr.train_fn()


def main(argv=None):

    if FLAGS.train:
        train_2()
    if FLAGS.eval:
        pass

if __name__ == '__main__':
    tf.app.run()
