import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


##### REQUIRED FLAGS #####
# Train or evaluate on a dataset?
tf.app.flags.DEFINE_bool('train', False, """Train model""")
tf.app.flags.DEFINE_bool('eval', False, """Evaluate model""")

# Data directories
tf.app.flags.DEFINE_string('input_dir',
                           None,
                           """Directory with all input files""")
tf.app.flags.DEFINE_string('eval_input_dir',
                           None,
                           """Directory with test files""")
# Expected Ising size
tf.app.flags.DEFINE_integer('ising_l',  None, """Ising model L to train with""")
# Epoch size of data
tf.app.flags.DEFINE_integer('epoch_size', None,
                            """Training Epoch Size""")
###########

### Other options. Some of these don't do anything yet

tf.app.flags.DEFINE_integer('max_steps', None,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('eval_steps', None,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 0, """The batch size.""")

tf.app.flags.DEFINE_string('train_dir',
                           './train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('eval_epoch_size', 0,
                            """Testing epoch size""")
tf.app.flags.DEFINE_string('eval_dir', './eval',
                           """Directory where to write event logs.""")

#
#tf.app.flags.DEFINE_integer('model_l', None, """Ising L for the model""")
tf.app.flags.DEFINE_bool('adam_optimizer', False, """Whether to use Adam""")


tf.app.flags.DEFINE_bool('clear_training', False, """Delete checkpoint files""")
tf.app.flags.DEFINE_bool('resume_training', False, """resume training from checkpoint""")


#  A struct for important configurations, read from a dictionary
class GlobalConfig:
    def __init__(self, d: dict = None):
        if d is not None:
            self.batch_size = d['batch_size']
            self.max_steps = d['max_steps']
            self.eval_steps = d['eval_steps']
            self.eval_freq = d['eval_freq']
            self.sum_save_freq = d['sum_save_freq']
            self.num_temps = d['num_temps']
            self.tc_partition = d.get('t_part', 0)
            self.log_freq = None
            if 'log_freq' in d:
                self.log_freq = d['log_freq']