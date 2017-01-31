"""
Trains and runs a fully connected model for inference on the
phase of the Ising ferromagnet

See e.g. Carrasquilla and Melko
 arXiv:1605.01735

"""

from _global_flags import *

from nets.ising_2d import fully_connected
from fileio import IsingFileRecord
from training import optm
from training import train_model
import tensorflow as tf
import numpy as np

#This is the temperature set used in MC simulations of the dataset
T_ARR_1 = [3., 2.95, 2.9, 2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55, 2.5, 2.45, 2.4,
2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.05, 2., 1.95, 1.9, 1.85, 1.8,
1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2,
1.15, 1.1, 1.05, 1.]

# dictionary of important settings
settings = {
    'batch_size':   128,  #number of ising configurations in a (mini) batch
    'max_steps':       20000,  # The number of batches to use in training
    'eval_steps':  32,  # How many batches to use to evaluate accuracy
    'eval_freq': 500,   # frequency at which to evaluate accuracy during training
    'sum_save_freq': 100, # frequency at which to save a checkpoint
    'num_temps': len(T_ARR_1),

    'enqueue_frac': 0.4  # fraction of epoch size that should be enqueued at all times. See IsingFileRecord
}
hparams = {
    'learn_rate': 0.05,
    'fc_size':  64,
    'momentum': 0.5
}


DEFAULT_TRAIN_INPUT_DIR = '/Users/hmunozbauza/Google Drive/C++ Projects/MonteCarlo-0.2/outputs/f2l10_2'
DEFAULT_TEST_OUTPUT_DIR = '/Users/hmunozbauza/Google Drive/C++ Projects/MonteCarlo-0.2/outputs/f2l10_val'


def train():
    # Initialize a struct with settings
    global_config = GlobalConfig(settings)

    ising_l = FLAGS.ising_l
    epoch_size = FLAGS.epoch_size
    # Minimum queue size for IsingFileRecord
    min_q_examples = int(settings['enqueue_frac']*epoch_size)
    # batch size flag overrides the dictionary value
    if FLAGS.batch_size > 0:
        global_config.batch_size = FLAGS.batch_size

    g = tf.Graph()
    with g.as_default():
        # Initialize a global step tensor
        global_step = train_model.make_global_step()

        # Create the training operation
        with tf.name_scope('training'):
            # Training batch from training dataset
            batch = IsingFileRecord.ising_2d_input(
                FLAGS.input_dir, global_config.batch_size,
                min_q_examples, ising_l)
            logits = fully_connected(
                batch.state, fc_size=hparams['fc_size'], training=True)
            # Loss function (x entropy) and training step
            loss = optm.loss_function(logits, batch.label)
            train_step = optm.make_train_op(loss, global_step, hparams)
        # Lets us reload any variables (respects variable scopes, not name scopes)
        tf.get_variable_scope().reuse_variables()
        # Create the accuracy evaluation op
        with tf.name_scope('testing'):
            test_batch = IsingFileRecord.ising_2d_input(
                FLAGS.eval_input_dir, global_config.batch_size,
                min_q_examples, ising_l)
            eval_logits = fully_connected(test_batch.state,
                                          hparams['fc_size'])
        # Run the training loop
        results = train_model.train_and_test_model(
            train_step, eval_logits, global_step, test_batch, global_config)

    # Print final results
    arr = np.array([T_ARR_1, results.pct_below_tc, results.t_accuracies]).transpose()
    print('*** RESULT *** ')
    for v in arr:
        print("{:.3}: {:.2%} at {:.2%}".format(
            v[0], v[1], v[2]))


## Run the program if executing this file


def main(argv=None):
    if FLAGS.train:
        train()

if __name__ == '__main__':
    tf.app.run()
